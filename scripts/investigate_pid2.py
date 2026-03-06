"""Investigate blood_sample / image_set / path_pid cardinality."""

import os
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
        print(f"\n{'='*70}\n{title}\n{'='*70}")
    if not rows:
        print("  (no rows)")
        return
    keys = list(rows[0].keys())
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print("  ".join(str(k).ljust(col_widths[k]) for k in keys))
    print("  ".join("-" * col_widths[k] for k in keys))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in keys))
    print(f"\n({len(rows)} rows)")


def main():
    with get_connection(DB_CONFIG) as conn:

        # Q1: blood_samples that map to more than one path_pid
        q1 = """
        SELECT
            bs.id AS bs_id,
            ss.name AS ss_name,
            COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)) AS n_path_pids,
            COUNT(DISTINCT s.id) AS n_image_sets,
            GROUP_CONCAT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)
                         ORDER BY 1 SEPARATOR ' | ') AS path_pids
        FROM blood_samples bs
        JOIN image_sets s ON s.id_blood_sample = bs.id
        JOIN image_tiles t ON t.id_image_set = s.id
        JOIN sample_sets ss ON bs.id_sample_set = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY bs.id, ss.name
        HAVING n_path_pids > 1
        ORDER BY n_path_pids DESC
        LIMIT 30
        """
        rows = run_query(conn, q1)
        print_table(rows, "Q1: blood_samples with >1 path_pid (Thin/Thick/slide2 counted separately)")

        # Q2: image_sets that map to more than one path_pid
        q2 = """
        SELECT
            s.id AS image_set_id,
            bs.id AS bs_id,
            ss.name AS ss_name,
            COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)) AS n_path_pids,
            GROUP_CONCAT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)
                         ORDER BY 1 SEPARATOR ' | ') AS path_pids
        FROM image_sets s
        JOIN blood_samples bs ON s.id_blood_sample = bs.id
        JOIN image_tiles t ON t.id_image_set = s.id
        JOIN sample_sets ss ON bs.id_sample_set = ss.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
        GROUP BY s.id, bs.id, ss.name
        HAVING n_path_pids > 1
        ORDER BY n_path_pids DESC
        LIMIT 20
        """
        rows = run_query(conn, q2)
        print_table(rows, "Q2: image_sets with >1 path_pid")

        # Q3: distribution of n_path_pids per blood_sample
        q3 = """
        SELECT n_path_pids, COUNT(*) AS n_blood_samples
        FROM (
            SELECT bs.id,
                   COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)) AS n_path_pids
            FROM blood_samples bs
            JOIN image_sets s ON s.id_blood_sample = bs.id
            JOIN image_tiles t ON t.id_image_set = s.id
            JOIN objects_of_interest o ON o.id_image_tile = t.id
            JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
            WHERE loc.id IN (33, 35, 37)
            GROUP BY bs.id
        ) sub
        GROUP BY n_path_pids ORDER BY n_path_pids
        """
        rows = run_query(conn, q3)
        print_table(rows, "Q3: Distribution — n_path_pids per blood_sample")

        # Q4: distribution of n_path_pids per image_set
        q4 = """
        SELECT n_path_pids, COUNT(*) AS n_image_sets
        FROM (
            SELECT s.id,
                   COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1)) AS n_path_pids
            FROM image_sets s
            JOIN image_tiles t ON t.id_image_set = s.id
            JOIN objects_of_interest o ON o.id_image_tile = t.id
            JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
            WHERE loc.id IN (33, 35, 37)
            GROUP BY s.id
        ) sub
        GROUP BY n_path_pids ORDER BY n_path_pids
        """
        rows = run_query(conn, q4)
        print_table(rows, "Q4: Distribution — n_path_pids per image_set")

        # Q5: strip _Thick/_Thin/_slide2 suffix — is the bare patient ID 1:1 with blood_sample?
        q5 = """
        SELECT n_bare_pids, COUNT(*) AS n_blood_samples
        FROM (
            SELECT bs.id,
                   COUNT(DISTINCT
                       REGEXP_REPLACE(
                           SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1),
                           '_(Thick|Thin|slide[0-9]+).*$', ''
                       )
                   ) AS n_bare_pids
            FROM blood_samples bs
            JOIN image_sets s ON s.id_blood_sample = bs.id
            JOIN image_tiles t ON t.id_image_set = s.id
            JOIN objects_of_interest o ON o.id_image_tile = t.id
            JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
            WHERE loc.id IN (33, 35, 37)
            GROUP BY bs.id
        ) sub
        GROUP BY n_bare_pids ORDER BY n_bare_pids
        """
        rows = run_query(conn, q5)
        print_table(rows, "Q5: After stripping _Thick/_Thin/_slideN — n_bare_pids per blood_sample (should all be 1)")

        # Q6: a few examples of blood_samples with >1 path_pid — show what the bare ID looks like
        q6 = """
        SELECT
            bs.id AS bs_id,
            SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1) AS path_pid,
            REGEXP_REPLACE(
                SUBSTRING_INDEX(REPLACE(t.file_location, '\\\\', '/'), '/', 1),
                '_(Thick|Thin|slide[0-9]+).*$', ''
            ) AS bare_pid,
            sm.name AS smear_type,
            s.id AS image_set_id
        FROM blood_samples bs
        JOIN image_sets s ON s.id_blood_sample = bs.id
        JOIN image_tiles t ON t.id_image_set = s.id
        JOIN smear_types sm ON s.id_smear_type = sm.id
        JOIN objects_of_interest o ON o.id_image_tile = t.id
        JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
        WHERE loc.id IN (33, 35, 37)
          AND bs.id IN (
              SELECT bs2.id
              FROM blood_samples bs2
              JOIN image_sets s2 ON s2.id_blood_sample = bs2.id
              JOIN image_tiles t2 ON t2.id_image_set = s2.id
              JOIN objects_of_interest o2 ON o2.id_image_tile = t2.id
              JOIN locator_algorithms loc2 ON o2.id_locator_algorithm = loc2.id
              WHERE loc2.id IN (33, 35, 37)
              GROUP BY bs2.id
              HAVING COUNT(DISTINCT SUBSTRING_INDEX(REPLACE(t2.file_location, '\\\\', '/'), '/', 1)) > 1
              LIMIT 5
          )
        GROUP BY bs.id, path_pid, bare_pid, sm.name, s.id
        ORDER BY bs.id, path_pid
        LIMIT 40
        """
        rows = run_query(conn, q6)
        print_table(rows, "Q6: Examples — blood_samples with >1 path_pid, showing bare_pid and smear_type")


if __name__ == "__main__":
    main()
