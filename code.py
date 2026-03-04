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
        lines.append(")")

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