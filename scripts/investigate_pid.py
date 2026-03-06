"""Investigate candidate patient identifiers (PID) in the autoscope DB.

Runs several exploratory SQL queries to determine which column is the correct
patient-level identifier to use as PID in the malaria pipeline.

Usage::

    DB_USER=xxx DB_PASSWORD=yyy uv run python scripts/investigate_pid.py

Or set DB_USER/DB_PASSWORD in your .env file and run::

    uv run python scripts/investigate_pid.py
"""

import os
import sys

import rootutils
from dotenv import load_dotenv

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

from src.data.utils.db_client import get_connection, run_query

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "database": os.environ.get("DB_NAME", "autoscope"),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}


def print_table(rows, title=""):
    if title:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
    if not rows:
        print("  (no rows)")
        return
    keys = list(rows[0].keys())
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(str(k).ljust(col_widths[k]) for k in keys)
    sep = "  ".join("-" * col_widths[k] for k in keys)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in keys))
    print(f"\n({len(rows)} rows)")


def main():
    with get_connection(DB_CONFIG) as conn:

        # ------------------------------------------------------------------
        # Q1: Sample of all candidate identifier columns side by side
        # ------------------------------------------------------------------
        q1 = """
        SELECT DISTINCT
            ss.id            AS ss_id,
            ss.name          AS ss_name,
            ss.description   AS ss_description,
            bs.id            AS bs_id,
            s.id             AS image_set_id,
            s.id_slide_number,
            SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1) AS path_pid,
            t.file_location
        FROM image_tiles t
        JOIN image_sets s      ON t.id_image_set    = s.id
        JOIN blood_samples bs  ON s.id_blood_sample = bs.id
        JOIN sample_sets ss    ON bs.id_sample_set  = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        LIMIT 50
        """
        rows = run_query(conn, q1)
        print_table(rows, "Q1: Sample of all candidate PID columns (50 rows)")

        # ------------------------------------------------------------------
        # Q2: Distinct sample_set names (current PID values)
        # ------------------------------------------------------------------
        q2 = """
        SELECT DISTINCT ss.id AS ss_id, ss.name AS ss_name, ss.description AS ss_description
        FROM sample_sets ss
        JOIN blood_samples bs ON bs.id_sample_set = ss.id
        JOIN image_sets s     ON s.id_blood_sample = bs.id
        JOIN image_tiles t    ON t.id_image_set = s.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        ORDER BY ss.id
        """
        rows = run_query(conn, q2)
        print_table(rows, "Q2: Distinct sample_sets (current PID source) in the dataset")

        # ------------------------------------------------------------------
        # Q3: Cardinality — how many blood samples / image sets per sample_set?
        # ------------------------------------------------------------------
        q3 = """
        SELECT
            ss.id,
            ss.name,
            COUNT(DISTINCT bs.id) AS n_blood_samples,
            COUNT(DISTINCT s.id)  AS n_image_sets,
            COUNT(DISTINCT t.id)  AS n_tiles
        FROM sample_sets ss
        JOIN blood_samples bs ON bs.id_sample_set  = ss.id
        JOIN image_sets s     ON s.id_blood_sample  = bs.id
        JOIN image_tiles t    ON t.id_image_set     = s.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY ss.id, ss.name
        ORDER BY ss.id
        """
        rows = run_query(conn, q3)
        print_table(rows, "Q3: Cardinality per sample_set (how many blood samples / image sets?)")

        # ------------------------------------------------------------------
        # Q4: file_location first segment vs ss.name
        # ------------------------------------------------------------------
        q4 = """
        SELECT
            ss.name   AS ss_name,
            SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1) AS path_pid,
            COUNT(DISTINCT t.id) AS n_tiles
        FROM image_tiles t
        JOIN image_sets s     ON t.id_image_set    = s.id
        JOIN blood_samples bs ON s.id_blood_sample = bs.id
        JOIN sample_sets ss   ON bs.id_sample_set  = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY ss.name, path_pid
        ORDER BY ss.name, path_pid
        """
        rows = run_query(conn, q4)
        print_table(rows, "Q4: ss.name vs file_location first segment (path_pid)")

        # ------------------------------------------------------------------
        # Q5: How many distinct path_pid values per ss.name? (1:1 or 1:N?)
        # ------------------------------------------------------------------
        q5 = """
        SELECT
            ss.name AS ss_name,
            COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)) AS n_distinct_path_pids
        FROM image_tiles t
        JOIN image_sets s     ON t.id_image_set    = s.id
        JOIN blood_samples bs ON s.id_blood_sample = bs.id
        JOIN sample_sets ss   ON bs.id_sample_set  = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY ss.name
        ORDER BY n_distinct_path_pids DESC, ss.name
        """
        rows = run_query(conn, q5)
        print_table(rows, "Q5: How many distinct path_pid values per ss.name? (>1 = path is more granular)")

        # ------------------------------------------------------------------
        # Q6: How many distinct ss.name values per path_pid? (1:1 or N:1?)
        # ------------------------------------------------------------------
        q6 = """
        SELECT
            SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1) AS path_pid,
            COUNT(DISTINCT ss.name) AS n_distinct_ss_names,
            COUNT(DISTINCT bs.id)   AS n_blood_samples,
            COUNT(DISTINCT s.id)    AS n_image_sets
        FROM image_tiles t
        JOIN image_sets s     ON t.id_image_set    = s.id
        JOIN blood_samples bs ON s.id_blood_sample = bs.id
        JOIN sample_sets ss   ON bs.id_sample_set  = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY path_pid
        ORDER BY path_pid
        """
        rows = run_query(conn, q6)
        print_table(rows, "Q6: Per path_pid: how many ss.names / blood samples / image sets?")

        # ------------------------------------------------------------------
        # Q7: Slide numbers — unique per patient or more granular?
        # ------------------------------------------------------------------
        q7 = """
        SELECT
            ss.name AS ss_name,
            COUNT(DISTINCT s.id_slide_number) AS n_slide_numbers,
            GROUP_CONCAT(DISTINCT s.id_slide_number ORDER BY s.id_slide_number) AS slide_numbers
        FROM sample_sets ss
        JOIN blood_samples bs ON bs.id_sample_set  = ss.id
        JOIN image_sets s     ON s.id_blood_sample  = bs.id
        JOIN image_tiles t    ON t.id_image_set     = s.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY ss.name
        ORDER BY ss.name
        """
        rows = run_query(conn, q7)
        print_table(rows, "Q7: Slide numbers per sample_set (is slide_number a patient-level ID?)")


if __name__ == "__main__":
    main()
