"""Simple, directory-local SQL templating.

Templates can reference other ``.sql`` files in the *same directory* using
``{{name}}`` placeholders. Expansion rules:

- If file ``A.sql`` contains ``{{foo}}`` and there is a sibling file
  ``foo.sql`` in the same directory, ``{{foo}}`` is replaced with the
  contents of ``foo.sql``.
- Expansion is applied repeatedly until no more placeholders have matching
  files in that directory.
- If, after expansion, any ``{{name}}`` placeholders remain, a ValueError
  is raised to surface misconfigurations early.
"""

import os
import re


PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Za-z0-9_]+)\s*}}")


def _replace_placeholders_once(sql, base_dir):
    """Replace one layer of ``{{name}}`` placeholders using .sql files in base_dir.

    Args:
        sql: Current SQL string.
        base_dir: Directory where referenced ``.sql`` files live.

    Returns:
        Tuple ``(new_sql, changed)`` where ``changed`` is True if any
        replacements were made.
    """
    names = sorted(set(PLACEHOLDER_PATTERN.findall(sql)))
    if not names:
        return sql, False

    result = sql
    changed = False

    for name in names:
        filename = f"{name}.sql"
        candidate = os.path.join(base_dir, filename)
        if not os.path.exists(candidate):
            continue

        with open(candidate) as f:
            contents = f.read()

        pattern = re.compile(r"{{\s*" + re.escape(name) + r"\s*}}")
        result = pattern.sub(contents, result)
        changed = True

    return result, changed


def render_sql(sql, query_path=None):
    """Expand ``{{name}}`` placeholders by inlining sibling ``.sql`` files.

    Args:
        sql: SQL string to render.
        query_path: Path to the ``.sql`` file this string came from. Placeholders
            are resolved relative to this file's directory.

    Returns:
        Rendered SQL string.

    Raises:
        ValueError: If, after expansion, any ``{{name}}`` placeholders remain.
    """
    if "{{" not in sql or query_path is None:
        return sql

    base_dir = os.path.dirname(os.path.abspath(query_path))
    text = sql

    while True:
        text, changed = _replace_placeholders_once(text, base_dir)
        if not changed:
            break

    remaining = sorted(set(PLACEHOLDER_PATTERN.findall(text)))
    if remaining:
        names = ", ".join(remaining)
        raise ValueError(f"Unresolved SQL placeholders: {names}")

    return text
