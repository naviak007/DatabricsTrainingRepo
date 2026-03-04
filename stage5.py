"""
=============================================================
SAS to PySpark Accelerator — Stage 5: Post-Processor
=============================================================
INPUT  : Raw PySpark code string from Stage 4
OUTPUT : Clean, formatted .py file  →  hr_report_stage5_final.py

Run order:
  stage1  →  stage2  →  stage3  →  stage4  →  stage5_postprocessor.py

Sub-steps:
  1. Receive raw code + metadata from Stage 4
  2. Build file header (source, date, block count, TODO count)
  3. Deduplicate and sort imports with isort
  4. Inject SparkSession block if missing
  5. Count TODO markers
  6. Format with black (PEP 8, line length 88)
  7. Write final .py file to disk
  8. Log metadata for Stage 6
=============================================================
"""

import os
import re
import ast
import sys
import json
import logging
import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage5")


from stage2_tokenizer import run_stage2
from stage3_parser    import run_stage3
from stage4_translator import run_stage4


# ══════════════════════════════════════════════════════════════
# STEP 2 — Build file header
# ══════════════════════════════════════════════════════════════

def build_header(source_file: str, block_count: int, todo_count: int) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 61
    status = "✅ All converted" if todo_count == 0 \
             else f"⚠️  {todo_count} TODO(s) need manual review"
    return (
        f"# {sep}\n"
        f"# SAS-to-PySpark Accelerator  v1.0\n"
        f"# Source  : {os.path.basename(source_file)}\n"
        f"# Date    : {now}\n"
        f"# Blocks  : {block_count} converted  |  {status}\n"
        f"# {sep}\n"
    )


# ══════════════════════════════════════════════════════════════
# STEP 3 — Deduplicate and sort imports
# Uses isort if available, otherwise falls back to sorted()
# ══════════════════════════════════════════════════════════════

def clean_imports(raw_code: str) -> str:
    """
    Extract all import lines from the code, deduplicate,
    and sort them (stdlib → third-party → local).
    Returns code with a clean import block at the top.
    """
    lines        = raw_code.splitlines()
    import_lines = []
    other_lines  = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(stripped)
        else:
            other_lines.append(line)

    # Deduplicate while preserving order
    seen   = set()
    unique = []
    for imp in import_lines:
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)

    # Sort: pyspark.sql.SparkSession  →  pyspark.sql  →  pyspark.sql.functions
    def sort_key(imp):
        if "SparkSession" in imp:
            return (0, imp)
        if "functions as F" in imp:
            return (1, imp)
        return (2, imp)

    sorted_imports = sorted(unique, key=sort_key)

    # Try isort for professional grouping
    try:
        import isort
        import_block = isort.code("\n".join(sorted_imports) + "\n").strip()
        log.info("  [Step 3] Imports sorted with isort")
    except ImportError:
        import_block = "\n".join(sorted_imports)
        log.info("  [Step 3] Imports sorted (isort not installed, using fallback)")

    body = "\n".join(other_lines).lstrip("\n")
    return import_block + "\n\n" + body


# ══════════════════════════════════════════════════════════════
# STEP 4 — Inject SparkSession if missing
# ══════════════════════════════════════════════════════════════

def inject_spark_session(code: str, app_name: str) -> str:
    if "SparkSession" not in code or "getOrCreate()" not in code:
        spark_block = (
            f'\nspark = (\n'
            f'    SparkSession.builder\n'
            f'    .appName("{app_name}")\n'
            f'    .getOrCreate()\n'
            f')\n'
        )
        # Insert after last import line
        lines  = code.splitlines()
        insert = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert = i + 1
        lines.insert(insert, spark_block)
        log.info("  [Step 4] SparkSession block injected")
        return "\n".join(lines)
    log.info("  [Step 4] SparkSession already present — skipped")
    return code


# ══════════════════════════════════════════════════════════════
# STEP 5 — Count TODO markers
# ══════════════════════════════════════════════════════════════

def count_todos(code: str) -> list:
    todos = []
    for i, line in enumerate(code.splitlines(), start=1):
        if re.search(r"#\s*TODO", line, re.IGNORECASE):
            todos.append((i, line.strip()))
    return todos


# ══════════════════════════════════════════════════════════════
# STEP 6 — Format with black
# ══════════════════════════════════════════════════════════════

def format_code(code: str) -> str:
    try:
        import black
        formatted = black.format_str(code, mode=black.Mode(line_length=88))
        log.info("  [Step 6] Code formatted with black (line_length=88)")
        return formatted
    except ImportError:
        log.info("  [Step 6] black not installed — skipping formatter")
        return code
    except Exception as e:
        log.warning(f"  [Step 6] black formatting failed: {e} — keeping raw code")
        return code


# ══════════════════════════════════════════════════════════════
# STEP 7 + 8 — run_stage5() orchestrator
# ══════════════════════════════════════════════════════════════

def run_stage5(
    raw_code:        str,
    source_file:     str,
    block_count:     int,
    todo_count:      int,
    out_dir:         str,
    output_filename: str = None,
) -> dict:
    """
    Clean, enrich, format, and write the final PySpark script.

    Args:
        raw_code    : PySpark code string from Stage 4
        source_file : Original .sas filename (for header)
        block_count : Number of SAS blocks converted
        todo_count  : Number of TODO flags from Stage 4
        out_dir     : Directory to write the final .py file

    Returns:
        metadata dict for Stage 6
    """
    log.info("=" * 55)
    log.info("STAGE 5 — Post-Processor")
    log.info("=" * 55)

    app_name = Path(source_file).stem  # hr_report

    # Step 2: Header
    header = build_header(source_file, block_count, todo_count)
    log.info("  [Step 2] File header built")

    # Step 3: Clean imports
    code = clean_imports(raw_code)
    log.info("  [Step 3] Imports deduplicated and sorted")

    # Step 4: SparkSession
    code = inject_spark_session(code, app_name)

    # Step 5: Count TODOs
    todos = count_todos(code)
    if todos:
        for line_no, text in todos:
            log.warning(f"  [Step 5] TODO on line {line_no}: {text}")
    else:
        log.info("  [Step 5] No TODO markers — all blocks converted")

    # Step 6: Format
    full_code = header + "\n" + code
    full_code = format_code(full_code)

    # Step 7: Write file
    if output_filename:
        out_path = Path(out_dir) / output_filename
    else:
        out_path = Path(out_dir) / f"{app_name}_stage5_final.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_code, encoding="utf-8")
    file_size = out_path.stat().st_size
    log.info(f"  [Step 7] Final script written: {out_path}  ({file_size} bytes)")

    # Step 8: Metadata for Stage 6
    metadata = {
        "source_file":  source_file,
        "output_file":  str(out_path),
        "block_count":  block_count,
        "todo_count":   len(todos),
        "todos":        [{"line": l, "text": t} for l, t in todos],
        "file_size":    file_size,
        "timestamp":    datetime.datetime.now().isoformat(),
    }

    log.info("=" * 55)
    log.info(f"Stage 5 complete  |  {len(todos)} TODOs  |  {file_size} bytes")
    log.info("=" * 55)

    return metadata, full_code


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import sys
    from pathlib import Path
    from stage1_preprocessor import run_stage1
    from stage2_tokenizer import run_stage2
    from stage3_parser import run_stage3
    from stage4_translator import run_stage4

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python stage5_postprocessor.py <input_file.sas>\n")
        sys.exit(1)

    input_sas = sys.argv[1]
    input_path = Path(input_sas)

    # ── Run full pipeline up to Stage 4 ─────────────────
    log.info("Running Stage 1...")
    stage1_result = run_stage1(input_sas)

    log.info("Running Stage 2...")
    token_map = run_stage2(stage1_result.blocks)

    log.info("Running Stage 3...")
    ast_nodes = run_stage3(token_map)

    log.info("Running Stage 4...")
    raw_code, todo_count = run_stage4(ast_nodes)

    # ── Stage 5 ─────────────────────────────────────────
    output_filename = f"{input_path.stem}_Converted.py"

    metadata, final_code = run_stage5(
        raw_code        = raw_code,
        source_file     = input_sas,
        block_count     = len(ast_nodes),
        todo_count      = todo_count,
        out_dir         = str(input_path.parent),
        output_filename = output_filename,
    )

    print(f"\n{'='*60}")
    print("  STAGE 5 RESULTS — Final PySpark Script")
    print(f"{'='*60}\n")
    print(final_code)

    # Save metadata JSON next to output file
    meta_path = input_path.parent / f"{input_path.stem}_stage5_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Metadata saved to: {meta_path}")