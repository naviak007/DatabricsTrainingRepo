"""
=============================================================
SAS to PySpark Accelerator — Stage 4: Translator
Expression AST Code Generator + PROC Translators
=============================================================
"""

import logging
import re

from stage3_parser import (
    LiteralNode,
    IdentifierNode,
    UnaryOpNode,
    BinaryOpNode,
    FunctionCallNode,
    InOpNode,
    BetweenOpNode,
    DataStepNode,
    ProcSortNode,
    ProcMeansNode,
    ProcSqlNode,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stage4")


# =============================================================
# SAS → Spark Function Map
# =============================================================

FUNCTION_MAP = {
    "UPCASE": "upper",
    "LOWCASE": "lower",
    "ABS": "abs",
    "ROUND": "round",
}


# =============================================================
# Expression Generator
# =============================================================

def generate_expr(node, imports):

    if isinstance(node, LiteralNode):
        return repr(node.value)

    if isinstance(node, IdentifierNode):
        imports.add("from pyspark.sql.functions import col")
        return f'col("{node.name}")'

    if isinstance(node, UnaryOpNode):

        operand = generate_expr(node.operand, imports)

        if node.operator == "NOT":
            return f"(~{operand})"

        return f"(-{operand})"

    if isinstance(node, BinaryOpNode):

        left = generate_expr(node.left, imports)
        right = generate_expr(node.right, imports)

        OP_MAP = {
            "=": "==",
            "EQ": "==",
            "NE": "!=",
            "^=": "!=",
            "GT": ">",
            "LT": "<",
            "GE": ">=",
            "LE": "<=",
            "AND": "&",
            "OR": "|",
        }

        op = OP_MAP.get(node.operator, node.operator)

        return f"({left} {op} {right})"

    if isinstance(node, InOpNode):

        expr = generate_expr(node.expr, imports)
        values = ", ".join(generate_expr(v, imports) for v in node.values)

        return f"{expr}.isin({values})"

    if isinstance(node, BetweenOpNode):

        expr = generate_expr(node.expr, imports)
        low = generate_expr(node.low, imports)
        high = generate_expr(node.high, imports)

        return f"(({expr} >= {low}) & ({expr} <= {high}))"

    if isinstance(node, FunctionCallNode):

        name = node.name.upper()

        if name == "MISSING":
            expr = generate_expr(node.args[0], imports)
            return f"{expr}.isNull()"

        spark_fn = FUNCTION_MAP.get(name)

        if spark_fn:

            imports.add(f"from pyspark.sql.functions import {spark_fn}")

            args = ", ".join(generate_expr(a, imports) for a in node.args)

            return f"{spark_fn}({args})"

    raise ValueError(f"Unsupported expression node {node}")


# =============================================================
# PROC SQL → DataFrame Translator
# =============================================================

def translate_sql_to_df(node, imports):

    sql = node.raw_sql.strip()

    imports.add("from pyspark.sql.functions import col")

    # SELECT
    select_match = re.search(
        r"SELECT\s+(.*?)\s+FROM",
        sql,
        re.IGNORECASE | re.DOTALL
    )

    # FROM table alias
    from_match = re.search(
        r"FROM\s+([a-zA-Z0-9_]+)\s*(\w+)?",
        sql,
        re.IGNORECASE
    )

    # JOIN table alias
    join_match = re.search(
        r"JOIN\s+([a-zA-Z0-9_]+)\s*(\w+)?",
        sql,
        re.IGNORECASE
    )

    # ON condition
    on_match = re.search(
        r"ON\s+(.*?)\s+(WHERE|GROUP|ORDER|$)",
        sql,
        re.IGNORECASE | re.DOTALL
    )

    # WHERE clause
    where_match = re.search(
        r"WHERE\s+(.*)",
        sql,
        re.IGNORECASE | re.DOTALL
    )

    if not select_match or not from_match:
        raise ValueError("Unsupported SQL structure")

    columns = select_match.group(1).strip()

    left_table = from_match.group(1)
    left_df = f"df_{left_table}"

    lines = []

   # ------------------------------------------------
   # JOIN CASE
   # ------------------------------------------------

    if join_match and on_match:

        right_table = join_match.group(1)
        right_df = f"df_{right_table}"

        on_condition = on_match.group(1).strip().rstrip(";")

        # Split join condition (a.id = b.id)
        left_col, right_col = [c.strip() for c in on_condition.split("=")]

        lines.append("df_sql_result = (")

        lines.append(
            f'    {left_df}.join({right_df}, col("{left_col}") == col("{right_col}"))')

    else:

        lines.append("df_sql_result = (")
        lines.append(f"    {left_df}")

        # ------------------------------------------------
        # SELECT
        # ------------------------------------------------

        if columns != "*":

            cols = [c.strip() for c in columns.split(",")]

            select_cols = ", ".join(f'"{c}"' for c in cols)

            lines.append(f"    .select({select_cols})")

        # ------------------------------------------------
        # WHERE
        # ------------------------------------------------

        if where_match:

            condition = where_match.group(1).strip().rstrip(";")

            lines.append(f'    .filter("{condition}")')

        lines.append(")")

        return "\n".join(lines)


# =============================================================
# Stage 4 Translator
# =============================================================

def run_stage4(ast_nodes):

    imports = set()
    sections = []

    for node in ast_nodes:

        # =====================================================
        # DATA STEP
        # =====================================================

        if isinstance(node, DataStepNode):

            out_df = f"df_{node.output_ds}"
            in_df = f"df_{node.input_ds}"

            lines = [f"{out_df} = {in_df}"]

            for where_expr in node.where:

                expr = generate_expr(where_expr, imports)

                lines.append(
                    f"    .filter({expr})"
                )

            for assign in node.assignments:

                rhs = generate_expr(assign.rhs, imports)

                lines.append(
                    f'    .withColumn("{assign.lhs}", {rhs})'
                )

            for if_node in node.if_then:

                cond = generate_expr(if_node.condition, imports)
                then_val = generate_expr(if_node.then_assign.rhs, imports)

                imports.add("from pyspark.sql.functions import when")

                if if_node.else_assign:

                    else_val = generate_expr(
                        if_node.else_assign.rhs,
                        imports
                    )

                    lines.append(
                        f'.withColumn("{if_node.then_assign.lhs}", '
                        f'when({cond}, {then_val}).otherwise({else_val}))'
                    )

                else:

                    lines.append(
                        f'.withColumn("{if_node.then_assign.lhs}", '
                        f'when({cond}, {then_val}).otherwise(None))'
                    )

            sections.append(" \\\n".join(lines))

        # =====================================================
        # PROC SORT
        # =====================================================

        elif isinstance(node, ProcSortNode):

            imports.add("from pyspark.sql.functions import col")

            out_df = f"df_{node.output_ds}"
            in_df = f"df_{node.input_ds}"

            order_expr = []

            for bv in node.by_vars:

                direction = ".desc()" if bv["desc"] else ".asc()"

                order_expr.append(
                    f'col("{bv["col"]}"){direction}'
                )

            code = (
                f"{out_df} = {in_df}.orderBy(\n"
                f"    {', '.join(order_expr)}\n"
                f")"
            )

            sections.append(code)

        # =====================================================
        # PROC MEANS
        # =====================================================

        elif isinstance(node, ProcMeansNode):

            imports.add("from pyspark.sql import functions as F")

            in_df = f"df_{node.input_ds}"
            out_df = "df_means_result"

            STAT_MAP = {
                "MEAN": "mean",
                "SUM": "sum",
                "MAX": "max",
                "MIN": "min",
                "STD": "stddev",
                "N": "count"
            }

            stats = node.stats if node.stats else ["MEAN"]

            agg_list = []

            for stat in stats:

                spark_fn = STAT_MAP.get(stat)

                for var in node.stat_vars:

                    agg_list.append(
                        f'F.{spark_fn}("{var}").alias("{stat.lower()}_{var}")'
                    )

            agg_expr = ", ".join(agg_list)

            if node.class_vars:

                group_cols = ", ".join(f'"{c}"' for c in node.class_vars)

                code = (
                    f"{out_df} = (\n"
                    f"    {in_df}\n"
                    f"    .groupBy({group_cols})\n"
                    f"    .agg({agg_expr})\n"
                    f")"
                )

            else:

                code = (
                    f"{out_df} = (\n"
                    f"    {in_df}\n"
                    f"    .agg({agg_expr})\n"
                    f")"
                )

            sections.append(code)

        # =====================================================
        # PROC SQL
        # =====================================================

        elif isinstance(node, ProcSqlNode):

            try:

                code = translate_sql_to_df(node, imports)

            except Exception:

                imports.add("from pyspark.sql import SparkSession")

                code = (
                    'df_sql_result = spark.sql("""\n'
                    f"{node.raw_sql}\n"
                    '""")'
                )

            sections.append(code)

    final_code = "\n".join(sorted(imports)) + "\n\n" + "\n\n".join(sections)

    return final_code, 0

    # =============================================================
# ENTRY POINT (Standalone Stage 4 Execution)
# =============================================================

if __name__ == "__main__":

    import sys
    from pathlib import Path

    from stage1_preprocessor import run_stage1
    from stage2_tokenizer import run_stage2
    from stage3_parser import run_stage3

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python stage4_translator.py <input_file.sas>\n")
        sys.exit(1)

    input_sas = sys.argv[1]
    input_path = Path(input_sas)

    # Stage 1
    log.info("Running Stage 1...")
    stage1_result = run_stage1(input_sas)

    # Stage 2
    log.info("Running Stage 2...")
    token_map = run_stage2(stage1_result.blocks)

    # Stage 3
    log.info("Running Stage 3...")
    ast_nodes = run_stage3(token_map)

    # Stage 4
    log.info("Running Stage 4...")
    pyspark_code, todo_count = run_stage4(ast_nodes)

    print("\n" + "=" * 60)
    print("STAGE 4 OUTPUT — Raw PySpark Code")
    print("=" * 60 + "\n")

    print(pyspark_code)

    # Save raw output
    output_path = input_path.parent / f"{input_path.stem}_stage4_raw.py"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pyspark_code)

    log.info(f"Raw PySpark saved to: {output_path}")

