"""
=============================================================
SAS to PySpark Accelerator — Stage 4: Translator (Full)
Expression AST Code Generator
=============================================================
"""

import logging
from stage3_parser import (
    LiteralNode,
    IdentifierNode,
    UnaryOpNode,
    BinaryOpNode,
    FunctionCallNode,
    InOpNode,
    BetweenOpNode,
    DataStepNode,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stage4")


FUNCTION_MAP = {
    "UPCASE": "upper",
    "LOWCASE": "lower",
    "ABS": "abs",
    "ROUND": "round",
}


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
# run_stage4 (Pipeline Compatible)
# =============================================================

def run_stage4(ast_nodes):

    imports = set()
    sections = []

    for node in ast_nodes:

        if isinstance(node, DataStepNode):
            out_df = f"df_{node.output_ds}"
            in_df = f"df_{node.input_ds}"

            lines = [f"{out_df} = {in_df}"]

            for where_expr in node.where:
                expr = generate_expr(where_expr, imports)
                lines.append(f"    .filter({expr})")

            for assign in node.assignments:
                rhs = generate_expr(assign.rhs, imports)
                lines.append(f'    .withColumn("{assign.lhs}", {rhs})')

            for if_node in node.if_then:
                cond = generate_expr(if_node.condition, imports)
                then_val = generate_expr(if_node.then_assign.rhs, imports)

                imports.add("from pyspark.sql.functions import when")

                if if_node.else_assign:
                    else_val = generate_expr(if_node.else_assign.rhs, imports)
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

    final_code = "\n".join(sorted(imports)) + "\n\n" + "\n\n".join(sections)

    return final_code, 0