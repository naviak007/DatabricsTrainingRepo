"""
=============================================================
SAS to PySpark Accelerator — Stage 2: Tokenizer (Lexer)
=============================================================
Takes the list of clean SAS blocks from Stage 1 and breaks
each one into a flat, ordered list of typed tokens.

Sub-steps:
  1. Define TokenType enum
  2. Build master compiled regex (named groups)
  3. Define SAS keyword set
  4. Iterate with re.finditer — classify each match
  5. KEYWORD vs IDENTIFIER disambiguation
  6. Parse numeric and string literals
  7. Track line numbers
  8. Append EOF token and return
=============================================================
"""

import re
import logging
from enum import Enum, auto
from collections import namedtuple
from typing import List
from dataclasses import dataclass, field

# ── import Stage 1 output types (when running together) ───────
# from stage1_preprocessor import SASBlock, PreProcessorResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("stage2")


# ══════════════════════════════════════════════════════════════
# STEP 1 — TokenType Enum
# Each member represents one category of SAS language token
# ══════════════════════════════════════════════════════════════

class TokenType(Enum):
    KEYWORD     = auto()   # DATA, SET, WHERE, PROC, RUN ...
    IDENTIFIER  = auto()   # variable names, dataset names
    OPERATOR    = auto()   # = > < >= <= ^= ne and or not
    LITERAL_STR = auto()   # 'hello'  "world"
    LITERAL_NUM = auto()   # 42  3.14  -7  1E6
    PUNCT       = auto()   # ; ( ) , . *
    EOF         = auto()   # end-of-block sentinel


# ── Token named tuple ─────────────────────────────────────────
# Lightweight, immutable object passed to Stage 3 (Parser)
Token = namedtuple("Token", ["type", "value", "line"])


# ══════════════════════════════════════════════════════════════
# STEP 2 — Master Regex  (named groups, most-specific first)
#
# Order matters:
#   • Multi-char operators (>=, <=, ^=) BEFORE single-char (>, <, =)
#   • Quoted strings BEFORE bare identifiers
#   • Numbers BEFORE bare words (avoids matching "3" as identifier)
# ══════════════════════════════════════════════════════════════

MASTER_PATTERN = re.compile(
    r"""
    (?P<LITERAL_STR>                    # --- Quoted strings ---
        '(?:[^'\\]|\\.)*'               #   single-quoted  'hello'
      | "(?:[^"\\]|\\.)*"               #   double-quoted  "world"
    )
    |
    (?P<LITERAL_NUM>                    # --- Numeric literals ---
        -?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?  # 42  3.14  -7  1.5E6
    )
    |
    (?P<OPERATOR>                       # --- Operators (multi-char FIRST) ---
        >=|<=|\^=|¬=|~=                 #   multi-char comparison
      | [+\-*/]                         #   arithmetic operators
      | [=<>]                           #   single-char comparison
      | \b(?:NE|LE|LT|GE|GT|EQ|AND|OR|NOT|IN|BETWEEN)\b  # word operators
    )
    |
    (?P<WORD>                           # --- Words (keyword or identifier) ---
        [A-Za-z_][A-Za-z0-9_]*         #   starts with letter or underscore
        (?:\.[A-Za-z_][A-Za-z0-9_]*)*  #   optional libname.dataset suffix
    )
    |
    (?P<PUNCT>                          # --- Punctuation ---
        [;(),.*]
    )
    |
    (?P<SKIP>                           # --- Whitespace / newlines (discard) ---
        \s+
    )
    """,
    re.VERBOSE | re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════
# STEP 3 — SAS Reserved Keyword Set
# Used to distinguish KEYWORD from IDENTIFIER for bare WORD matches
# ══════════════════════════════════════════════════════════════

SAS_KEYWORDS = {
    # DATA step
    "DATA", "SET", "MERGE", "UPDATE", "MODIFY",
    "BY", "KEEP", "DROP", "RENAME", "WHERE", "IF", "THEN",
    "ELSE", "DO", "END", "RETURN", "OUTPUT", "ARRAY",
    "RETAIN", "LENGTH", "FORMAT", "INFORMAT", "LABEL",
    "RUN",
    # PROC
    "PROC", "QUIT",
    # PROC names (treated as keywords when following PROC)
    "SQL", "SORT", "MEANS", "SUMMARY", "FREQ", "TABULATE",
    "REPORT", "PRINT", "CONTENTS", "DATASETS",
    # SQL
    "SELECT", "FROM", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
    "FULL", "ON", "AS", "CREATE", "TABLE", "INSERT", "INTO",
    "UPDATE", "DELETE", "GROUP", "ORDER", "HAVING", "DISTINCT",
    "UNION", "CASE", "WHEN", "THEN", "ELSE", "END",
    # Operators (word form)
    "AND", "OR", "NOT", "IN", "NE", "LE", "LT", "GE", "GT", "EQ",
    # Other common
    "CLASS", "VAR", "OUT", "NOPRINT", "DESCENDING", "NODUPKEY",
    "MEAN", "MAX", "MIN", "SUM", "STD", "N",
}


# ══════════════════════════════════════════════════════════════
# STEPS 4-8 — Tokenize one SAS block
# ══════════════════════════════════════════════════════════════

def tokenize_block(sas_text: str, block_index: int = 1) -> List[Token]:
    """
    Tokenize a single cleaned SAS block string into a list of Tokens.

    Steps performed:
      4. Iterate with re.finditer over the master pattern
      5. Disambiguate WORD matches → KEYWORD or IDENTIFIER
      6. Parse numeric and strip-quote string literals
      7. Track line number via newline count
      8. Append EOF sentinel and return

    Args:
        sas_text:    Cleaned SAS block string from Stage 1
        block_index: Block number (for logging only)

    Returns:
        List of Token(type, value, line) ending with EOF token
    """
    tokens   = []
    line_num = 1   # Step 7: track current line number

    for match in MASTER_PATTERN.finditer(sas_text):

        # Step 7: count newlines in whitespace/text before this match
        line_num += sas_text[: match.start()].count("\n") - \
                    sas_text[: match.start() - (match.start() - match.start())].count("\n") \
                    if tokens else sas_text[:match.start()].count("\n")
        # Simpler running approach: count newlines seen up to this point
        line_num = sas_text[:match.start()].count("\n") + 1

        # ── Step 4: identify which named group matched ────────
        kind = match.lastgroup

        # Discard whitespace
        if kind == "SKIP":
            continue

        raw = match.group()

        # ── Step 5: WORD → KEYWORD or IDENTIFIER ──────────────
        if kind == "WORD":
            if raw.upper() in SAS_KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, raw.upper(), line_num))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, raw, line_num))

        # ── Step 6a: String literals — strip surrounding quotes ─
        elif kind == "LITERAL_STR":
            value = raw[1:-1]   # remove the outer ' ' or " "
            tokens.append(Token(TokenType.LITERAL_STR, value, line_num))

        # ── Step 6b: Numeric literals — parse to float ─────────
        elif kind == "LITERAL_NUM":
            try:
                value = float(raw)
                # Store as int if it's a whole number (cleaner output)
                value = int(value) if value == int(value) else value
            except ValueError:
                value = raw
            tokens.append(Token(TokenType.LITERAL_NUM, value, line_num))

        # ── Operators ──────────────────────────────────────────
        elif kind == "OPERATOR":
            tokens.append(Token(TokenType.OPERATOR, raw.upper(), line_num))

        # ── Punctuation ────────────────────────────────────────
        elif kind == "PUNCT":
            tokens.append(Token(TokenType.PUNCT, raw, line_num))

    # Step 8: append EOF sentinel
    last_line = sas_text.count("\n") + 1
    tokens.append(Token(TokenType.EOF, None, last_line))

    log.info(f"  [Block {block_index}] Tokenized → {len(tokens) - 1} tokens  (+EOF)")
    return tokens


# ══════════════════════════════════════════════════════════════
# MAIN — Tokenize all blocks from Stage 1
# ══════════════════════════════════════════════════════════════

def run_stage2(blocks) -> dict:
    """
    Tokenize every SAS block produced by Stage 1.

    Args:
        blocks: List of SASBlock objects from stage1.run_stage1()

    Returns:
        dict mapping block.index → List[Token]
    """
    log.info("=" * 55)
    log.info("STAGE 2 — Tokenizer")
    log.info("=" * 55)

    token_map = {}
    for block in blocks:
        log.info(f"  Tokenizing Block {block.index}: {block.block_type}")
        tokens = tokenize_block(block.raw_text, block.index)
        token_map[block.index] = tokens

    total = sum(len(t) - 1 for t in token_map.values())   # -1 to exclude EOF
    log.info("=" * 55)
    log.info(f"Stage 2 complete  |  {len(token_map)} blocks  |  {total} tokens total")
    log.info("=" * 55)

    return token_map


# ══════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════

def print_tokens(token_map: dict, blocks=None, max_tokens: int = 60):
    """Print a readable token listing for each block."""

    block_lookup = {b.index: b for b in blocks} if blocks else {}

    print(f"\n{'='*60}")
    print("  STAGE 2 RESULTS — Token Listing")
    print(f"{'='*60}")

    for block_idx, tokens in token_map.items():
        block_type = block_lookup[block_idx].block_type if block_lookup else f"Block {block_idx}"
        print(f"\n[Block {block_idx}]  {block_type}")
        print(f"  {'-'*56}")
        print(f"  {'TYPE':<18} {'VALUE':<30} LINE")
        print(f"  {'-'*56}")

        display = tokens[:max_tokens]
        for tok in display:
            val = str(tok.value) if tok.value is not None else "(None)"
            print(f"  {tok.type.name:<18} {val:<30} {tok.line}")

        if len(tokens) > max_tokens:
            print(f"  ... ({len(tokens) - max_tokens} more tokens not shown)")

    print(f"\n{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT — runs Stage 1 then Stage 2
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python stage2_tokenizer.py <input_file.sas>\n")
        sys.exit(1)

    input_sas = sys.argv[1]

    # Import Stage 1 normally (assumes same folder / package structure)
    from stage1_preprocessor import run_stage1

    print("\nRunning Stage 1 to generate clean blocks...")
    stage1_result = run_stage1(input_sas)

    # Run Stage 2
    token_map = run_stage2(stage1_result.blocks)

    # Print results
    print_tokens(token_map, blocks=stage1_result.blocks)

    # Save token listing next to input file
    input_path = Path(input_sas)
    output_path = input_path.parent / f"{input_path.stem}_stage2_tokens.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for block_idx, tokens in token_map.items():
            block_type = stage1_result.blocks[block_idx - 1].block_type
            f.write(f"\n[Block {block_idx}] {block_type}\n")
            f.write(f"  {'TYPE':<18} {'VALUE':<30} LINE\n")
            f.write(f"  {'-'*56}\n")
            for tok in tokens:
                val = str(tok.value) if tok.value is not None else "(None)"
                f.write(f"  {tok.type.name:<18} {val:<30} {tok.line}\n")

    log.info(f"Token listing saved to: {output_path}")