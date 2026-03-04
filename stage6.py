"""
=============================================================
SAS to PySpark Accelerator — Stage 6: Validator
=============================================================
Runs full pipeline:
stage1 → stage2 → stage3 → stage4 → stage5 → stage6

Final output:
    - Generated PySpark file
    - Validation JSON report
    - Console PASS/WARN/FAIL summary
=============================================================
"""

import os
import re
import ast
import sys
import json
import logging
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage6")

# ─────────────────────────────────────────
# Adjust this path to your accelerator folder
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────
# Import all pipeline stages
# ─────────────────────────────────────────
from stage1_preprocessor import run_stage1
from stage2_tokenizer import run_stage2
from stage3_parser import run_stage3
from stage4_translator import run_stage4
from stage5_postprocessor import run_stage5


# ══════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    file: str
    status: str
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = ""

    def add(self, name: str, status: str, detail: str):
        self.checks.append(CheckResult(name, status, detail))
        if status == "FAIL":
            self.status = "FAIL"
        elif status == "WARN" and self.status == "PASS":
            self.status = "WARN"

    def to_dict(self):
        return {
            "file": self.file,
            "status": self.status,
            "timestamp": self.timestamp,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


# ══════════════════════════════════════════════════════════════
# Stage 6 Core Logic
# ══════════════════════════════════════════════════════════════

def run_stage6(py_filepath: str, out_dir: str) -> ValidationReport:

    log.info("=" * 55)
    log.info("STAGE 6 — Validator")
    log.info("=" * 55)

    report = ValidationReport(
        file=py_filepath,
        status="PASS",
        timestamp=datetime.datetime.now().isoformat(),
    )

    # Step 1: Read file
    path = Path(py_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Generated file not found: {py_filepath}")

    source = path.read_text(encoding="utf-8")
    log.info(f"  Read: {path.name}")

    # Step 2: Syntax check
    try:
        tree = ast.parse(source)
        report.add("Python syntax", "PASS", "No syntax errors")
    except SyntaxError as e:
        report.add("Python syntax", "FAIL", f"Line {e.lineno}: {e.msg}")
        return report

    # Step 3: TODO scan
    todos = []
    for i, line in enumerate(source.splitlines(), 1):
        if "TODO" in line.upper():
            todos.append(i)

    if todos:
        report.add("TODO markers", "WARN", f"Found on lines {todos}")
    else:
        report.add("TODO markers", "PASS", "No TODO markers found")

    # Step 4: Write JSON report
    report_path = Path(out_dir) / f"{path.stem}_stage6_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    log.info(f"Report written: {report_path}")
    log.info(f"Overall Status: {report.status}")
    log.info("=" * 55)

    return report


# ══════════════════════════════════════════════════════════════
# Pretty Print
# ══════════════════════════════════════════════════════════════

def print_report(report: ValidationReport):
    print("\n" + "=" * 60)
    print("STAGE 6 VALIDATION REPORT")
    print("=" * 60)
    print(f"File   : {os.path.basename(report.file)}")
    print(f"Status : {report.status}")
    print("-" * 60)

    for c in report.checks:
        print(f"{c.name:<25} {c.status:<6} {c.detail}")

    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT — FULL PIPELINE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python stage6_validator.py <input_file.sas>\n")
        sys.exit(1)

    input_sas = Path(sys.argv[1]).resolve()

    if not input_sas.exists():
        print(f"\nError: File not found -> {input_sas}\n")
        sys.exit(1)

    if input_sas.suffix.lower() != ".sas":
        print("\nError: Input must be a .sas file\n")
        sys.exit(1)

    log.info("=" * 60)
    log.info("SAS → PySpark Accelerator (Full Pipeline)")
    log.info("=" * 60)
    log.info(f"Input : {input_sas.name}")
    log.info(f"Folder: {input_sas.parent}")

    # ── Stage 1
    stage1_result = run_stage1(str(input_sas))

    # ── Stage 2
    token_map = run_stage2(stage1_result.blocks)

    # ── Stage 3
    ast_nodes = run_stage3(token_map)

    # ── Stage 4
    raw_code, todo_count = run_stage4(ast_nodes)

    # ── Stage 5
    output_filename = f"{input_sas.stem}_Converted.py"

    metadata, _ = run_stage5(
        raw_code        = raw_code,
        source_file     = str(input_sas),
        block_count     = len(ast_nodes),
        todo_count      = todo_count,
        out_dir         = str(input_sas.parent),
        output_filename = output_filename,
    )

    # ── Stage 6
    py_file = metadata["output_file"]
    report  = run_stage6(py_file, str(input_sas.parent))

    print_report(report)

    log.info("=" * 60)
    log.info("Pipeline completed successfully.")
    log.info("=" * 60)