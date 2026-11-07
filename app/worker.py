import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from .catalog import get_checklist_version, get_conversation
from .db import connect, database_path, initialize_database, is_database_busy
from .evaluation import EvaluationError, evaluate

MAX_ATTEMPTS = 3
MAX_RETRY_DELAY = 300
logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class Job:
    id: int
    conversation_id: int
    checklist_version_id: int
    attempts: int
    token: str


def _claim_next(path: Path, lease_seconds: float) -> Job | None:
    with closing(connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        connection.execute(
            """
            UPDATE evaluation_jobs
            SET status = 'failed', lease_token = NULL, lease_until = NULL,
                error = 'Worker stopped before completing the final attempt'
            WHERE status = 'running' AND lease_until <= ? AND attempts >= ?
            """,
            (now, MAX_ATTEMPTS),
        )
        row = connection.execute(
            """
            SELECT j.id, j.conversation_id, j.attempts, r.checklist_version_id
            FROM evaluation_jobs AS j
            JOIN evaluation_runs AS r ON r.id = j.run_id
            WHERE j.attempts < ? AND (
                (j.status = 'queued' AND j.available_at <= ?)
                OR (j.status = 'running' AND j.lease_until <= ?)
            )
            ORDER BY j.id
            LIMIT 1
            """,
            (MAX_ATTEMPTS, now, now),
        ).fetchone()
        if row is None:
            return None

        token = uuid4().hex
        connection.execute(
            """
            UPDATE evaluation_jobs
            SET status = 'running', attempts = attempts + 1, lease_token = ?,
                lease_until = ?, error = NULL
            WHERE id = ?
            """,
            (token, now + lease_seconds, row["id"]),
        )
        return Job(row["id"], row["conversation_id"], row["checklist_version_id"], row["attempts"] + 1, token)


def _complete(path: Path, job: Job, answers: list[dict]) -> None:
    answers_json = json.dumps(answers, ensure_ascii=False, allow_nan=False)
    with closing(connect(path)) as connection, connection:
        updated = connection.execute(
            """
            UPDATE evaluation_jobs
            SET status = 'completed', lease_token = NULL, lease_until = NULL, error = NULL
            WHERE id = ? AND status = 'running' AND lease_token = ?
            """,
            (job.id, job.token),
        )
        if updated.rowcount:
            connection.execute(
                "INSERT INTO evaluations (job_id, answers_json) VALUES (?, ?)",
                (job.id, answers_json),
            )


def _fail(
    path: Path,
    job: Job,
    message: str,
    *,
    retryable: bool = False,
    retry_after: float | None = None,
) -> None:
    retry = retryable and job.attempts < MAX_ATTEMPTS
    delay = min(MAX_RETRY_DELAY, max(2**job.attempts, retry_after or 0))
    with closing(connect(path)) as connection, connection:
        connection.execute(
            """
            UPDATE evaluation_jobs
            SET status = ?, available_at = ?, lease_token = NULL, lease_until = NULL, error = ?
            WHERE id = ? AND status = 'running' AND lease_token = ?
            """,
            (
                "queued" if retry else "failed",
                time.time() + delay if retry else 0,
                message.encode("utf-8", errors="replace").decode("utf-8"),
                job.id,
                job.token,
            ),
        )


def _retry_on_busy(operation: Callable[[], T]) -> T:
    delay = 0.05
    while True:
        try:
            return operation()
        except sqlite3.OperationalError as error:
            if not is_database_busy(error):
                raise
            if delay == 0.05:
                logger.warning("Waiting for a SQLite lock to clear")
            time.sleep(delay)
            delay = min(2 * delay, 0.5)


def _record_failure(
    path: Path,
    job: Job,
    message: str,
    *,
    retryable: bool = False,
    retry_after: float | None = None,
) -> None:
    try:
        _retry_on_busy(lambda: _fail(path, job, message, retryable=retryable, retry_after=retry_after))
    except Exception:
        logger.exception("Could not record failure for job %s; its lease will expire", job.id)


def _load_inputs(path: Path, job: Job) -> tuple[dict, dict]:
    with closing(connect(path)) as connection:
        conversation = get_conversation(connection, job.conversation_id)
        checklist = get_checklist_version(connection, job.checklist_version_id)
    if conversation is None or checklist is None:
        raise EvaluationError("Conversation or checklist version is missing")
    return conversation, checklist


def lease_seconds() -> float:
    try:
        timeout = float(os.environ.get("CQA_LLM_TIMEOUT", "60"))
    except ValueError:
        timeout = 60
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 60
    return max(30, 2 * timeout + 15)


def run_once(path: Path, *, lease_duration: float | None = None) -> bool:
    try:
        job = _claim_next(path, lease_duration if lease_duration is not None else lease_seconds())
    except sqlite3.OperationalError as error:
        if not is_database_busy(error):
            raise
        return False
    if job is None:
        return False

    try:
        conversation, checklist = _retry_on_busy(lambda: _load_inputs(path, job))
        answers = evaluate(conversation, checklist)
    except KeyboardInterrupt:
        try:
            _fail(path, job, "Worker stopped during evaluation", retryable=True)
        except sqlite3.Error:
            logger.warning("Could not release job %s during shutdown; its lease will expire", job.id)
        raise
    except EvaluationError as error:
        _record_failure(path, job, str(error), retryable=error.retryable, retry_after=error.retry_after)
    except Exception:
        logger.exception("Unexpected error while evaluating job %s", job.id)
        _record_failure(path, job, "Unexpected evaluation error")
    else:
        try:
            # Keep the computed answer until it can be committed or the job is reclaimed.
            _retry_on_busy(lambda: _complete(path, job, answers))
        except Exception:
            logger.exception("Could not save evaluation for job %s", job.id)
            _record_failure(path, job, "Could not save evaluation result")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = database_path()
    initialize_database(path)
    logger.info("Evaluation worker ready on %s", path)
    try:
        while True:
            if not run_once(path):
                time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Evaluation worker stopped")


if __name__ == "__main__":
    main()
