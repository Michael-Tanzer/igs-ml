"""MySQL database client for running queries with an external config."""

import os
from contextlib import contextmanager

import pymysql

from src.utils.sql_templates import render_sql


@contextmanager
def get_connection(config):
    """Context manager that yields a MySQL connection from resolved config.

    Args:
        config: Dict-like with host, port, database, user, password (e.g. from
            load_db_config() or Hydra cfg.db).

    Yields:
        pymysql connection. Closed on exit.
    """
    conn = pymysql.connect(
        host=config["host"],
        port=int(config["port"]),
        database=config["database"],
        user=config["user"],
        password=config["password"],
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        yield conn
    finally:
        conn.close()


def run_query(conn, query):
    """Execute a SQL query and return rows as list of dicts.

    Args:
        conn: pymysql connection (e.g. from get_connection()).
        query: SQL string to execute.

    Returns:
        List of dicts, one per row; keys are column names.
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def run_query_from_file(conn, path):
    """Execute a SQL query loaded from a file with {{name}} expansion.

    The path can be absolute or relative to the current working directory.
    Placeholders like ``{{foo}}`` will be resolved against ``foo.sql`` in the
    same directory as the file at ``path``.

    Args:
        conn: pymysql connection (e.g. from get_connection()).
        path: Filesystem path to a .sql file.

    Returns:
        List of dicts, one per row; keys are column names.
    """
    fs_path = os.path.abspath(path)
    with open(fs_path) as f:
        raw_sql = f.read()
    query = render_sql(raw_sql, query_path=fs_path)
    return run_query(conn, query)
