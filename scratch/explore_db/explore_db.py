"""Explore DB data from example_query.sql: run query, print shape/dtypes/stats and optional summary file.

Requires .env with DB_USER and DB_PASSWORD (see .env.example). Run from project root or with
rootutils so PROJECT_ROOT and .env are set (e.g. uv run python scratch/explore_db.py).
"""

import os

import click
import pandas as pd
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.db_client import (
    get_connection,
    run_query_from_file,
)


def _project_root():
    """Project root directory (set by rootutils or PROJECT_ROOT)."""
    return os.environ.get("PROJECT_ROOT", os.getcwd())


def _load_db_config():
    """Load DB config from YAML and resolve env interpolations."""
    root = _project_root()
    path = os.path.join(root, "configs", "db", "default.yaml")
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def _explore(df):
    """Build exploration summary: shape, dtypes, numeric describe, categorical value counts, nulls.

    Returns:
        Single string with full summary.
    """
    lines = []
    lines.append("=== Shape ===")
    lines.append(f"{df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")

    lines.append("=== Dtypes ===")
    lines.append(df.dtypes.to_string())
    lines.append("")

    numeric = df.select_dtypes(include=["number"])
    if not numeric.empty:
        lines.append("=== Numeric summary (describe) ===")
        lines.append(numeric.describe().to_string())
        lines.append("")

    categorical_cols = ["stage", "species", "locator_algorithm", "smear_type"]
    for col in categorical_cols:
        if col in df.columns:
            lines.append(f"=== Value counts: {col} ===")
            lines.append(df[col].value_counts(dropna=False).head(20).to_string())
            lines.append("")

    if "id_image_set" in df.columns:
        lines.append("=== Value counts: id_image_set ===")
        lines.append(df["id_image_set"].value_counts(dropna=False).to_string())
        lines.append("")
    if "PID" in df.columns:
        lines.append("=== Value counts: PID ===")
        lines.append(df["PID"].value_counts(dropna=False).head(20).to_string())
        lines.append("")

    lines.append("=== Null counts (key columns) ===")
    key_cols = ["stage", "species", "file_location", "parasitemia", "object_id"]
    for col in key_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            pct = 100.0 * n / len(df) if len(df) else 0
            lines.append(f"  {col}: {n} ({pct:.1f}%)")
    lines.append("")

    lines.append("=== Sample rows (first 5) ===")
    lines.append(df.head().to_string())
    return "\n".join(lines)


@click.command()
@click.option(
    "--query-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to SQL file. Default: PROJECT_ROOT/scratch/example_query.sql",
)
@click.option(
    "--output-summary",
    type=click.Path(),
    default=None,
    help="If set, write exploration summary to this file instead of stdout.",
)
def main(query_path, output_summary):
    """Run example query against DB and print exploration (shape, dtypes, stats, value counts, nulls)."""
    if query_path is None:
        query_path = os.path.join(_project_root(), "scratch", "example_query.sql")
    if not os.path.isfile(query_path):
        raise click.ClickException(f"Query file not found: {query_path}")
    cfg = _load_db_config()

    with get_connection(cfg) as conn:
        rows = run_query_from_file(conn, query_path)

    df = pd.DataFrame(rows)
    if df.empty:
        click.echo("Query returned 0 rows.")
        return

    summary = _explore(df)
    if output_summary:
        with open(output_summary, "w") as f:
            f.write(summary)
        click.echo(f"Summary written to {output_summary}")
    else:
        click.echo(summary)


if __name__ == "__main__":
    main()
