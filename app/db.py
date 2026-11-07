import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "cqa.sqlite3"


def database_path() -> Path:
    configured_path = os.environ.get("CQA_DB_PATH")
    return Path(configured_path or DEFAULT_DATABASE_PATH).expanduser().resolve()


def connect(path: Path, *, create: bool = False) -> sqlite3.Connection:
    mode = "rwc" if create else "rw"
    connection = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def is_database_busy(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and (code & 0xff) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


def initialize_database(path: Path) -> None:
    from .catalog import init_schema as initialize_catalog
    from .workflow import init_schema as initialize_workflow

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 20
    while True:
        try:
            with closing(connect(path, create=True)) as connection:
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    connection.execute("PRAGMA journal_mode = WAL").fetchone()
                initialize_catalog(connection)
                initialize_workflow(connection)
            return
        except sqlite3.OperationalError as error:
            if not is_database_busy(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


if __name__ == "__main__":
    path = database_path()
    initialize_database(path)
    print(f"SQLite ready: {path}")
