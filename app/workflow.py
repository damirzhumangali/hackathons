import json
import sqlite3
from contextlib import closing
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from .catalog import GroupID, NonEmptyText, PathID, get_checklist_version, get_conversation, list_conversation_ids
from .db import connect
from .evaluation import resolve_evidence, score, validate_answer

router = APIRouter(prefix="/api", tags=["Evaluations"])

SQLITE_MAX_ID = 2**63 - 1
PositiveID = Annotated[int, Field(strict=True, gt=0, le=SQLITE_MAX_ID)]
Answer = StrictBool | StrictInt | Literal["N/A"]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    checklist_version_id: PositiveID
    conversation_ids: list[PositiveID] | None = Field(default=None, min_length=1)
    group_id: GroupID | None = None

    @model_validator(mode="after")
    def check_selection(self):
        if (self.conversation_ids is None) == (self.group_id is None):
            raise ValueError("Provide either conversation_ids or group_id")
        return self


class AppealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    criterion_id: NonEmptyText
    comment: NonEmptyText = Field(max_length=4000)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal["confirm", "change"]
    reason: NonEmptyText = Field(max_length=4000)
    answer: Answer | None = None

    @model_validator(mode="after")
    def check_answer(self):
        if self.decision == "change" and self.answer is None:
            raise ValueError("A changed decision requires an answer")
        if self.decision == "confirm" and self.answer is not None:
            raise ValueError("A confirmed decision does not take an answer")
        return self


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY,
            checklist_version_id INTEGER NOT NULL REFERENCES checklist_versions(id),
            group_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS evaluation_jobs (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES evaluation_runs(id),
            conversation_id INTEGER NOT NULL REFERENCES conversation_versions(id),
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'completed', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL DEFAULT 0,
            lease_token TEXT,
            lease_until REAL,
            error TEXT,
            UNIQUE (run_id, conversation_id)
        );
        CREATE INDEX IF NOT EXISTS evaluation_job_queue
            ON evaluation_jobs (status, available_at, lease_until);
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY,
            job_id INTEGER NOT NULL UNIQUE REFERENCES evaluation_jobs(id),
            answers_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS appeals (
            id INTEGER PRIMARY KEY,
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
            criterion_id TEXT NOT NULL,
            comment TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
            decision TEXT CHECK (decision IN ('confirm', 'change')),
            reason TEXT,
            answer_json TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            resolved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS appeals_by_evaluation ON appeals (evaluation_id, id);
        CREATE UNIQUE INDEX IF NOT EXISTS appeals_one_open_per_criterion
            ON appeals (evaluation_id, criterion_id) WHERE status = 'open';
        """
    )


def _get_run(connection: sqlite3.Connection, run_id: int) -> dict:
    run = connection.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise HTTPException(404, "Evaluation run not found")
    jobs = [
        dict(row)
        for row in connection.execute(
            """
            SELECT j.id, j.conversation_id, j.status, j.attempts, j.error,
                   e.id AS evaluation_id
            FROM evaluation_jobs AS j
            LEFT JOIN evaluations AS e ON e.job_id = j.id
            WHERE j.run_id = ?
            ORDER BY j.id
            """,
            (run_id,),
        )
    ]
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "total": len(jobs)}
    for job in jobs:
        counts[job["status"]] += 1

    if all(job["attempts"] == 0 for job in jobs):
        status = "queued"
    elif counts["queued"] or counts["running"]:
        status = "running"
    elif counts["failed"] == len(jobs):
        status = "failed"
    elif counts["failed"]:
        status = "completed_with_errors"
    else:
        status = "completed"
    return {**dict(run), "status": status, "counts": counts, "jobs": jobs}


@router.post("/evaluation-runs", status_code=202)
def create_run(payload: RunRequest, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        if get_checklist_version(connection, payload.checklist_version_id) is None:
            raise HTTPException(404, "Checklist version not found")

        if payload.group_id is not None:
            conversation_ids = list_conversation_ids(connection, payload.group_id)
            if not conversation_ids:
                raise HTTPException(404, "No conversations found for this group")
        else:
            conversation_ids = list(dict.fromkeys(payload.conversation_ids or []))
            for conversation_id in conversation_ids:
                if get_conversation(connection, conversation_id) is None:
                    raise HTTPException(404, f"Conversation version {conversation_id} not found")

        run_id = connection.execute(
            "INSERT INTO evaluation_runs (checklist_version_id, group_id) VALUES (?, ?)",
            (payload.checklist_version_id, payload.group_id),
        ).lastrowid
        connection.executemany(
            "INSERT INTO evaluation_jobs (run_id, conversation_id) VALUES (?, ?)",
            [(run_id, conversation_id) for conversation_id in conversation_ids],
        )
        return _get_run(connection, run_id)


@router.get("/evaluation-runs/{run_id}")
def get_run(run_id: PathID, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN")
        return _get_run(connection, run_id)


def _evaluation_row(connection: sqlite3.Connection, evaluation_id: int) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT e.*, j.conversation_id, r.checklist_version_id
        FROM evaluations AS e
        JOIN evaluation_jobs AS j ON j.id = e.job_id
        JOIN evaluation_runs AS r ON r.id = j.run_id
        WHERE e.id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Evaluation not found")
    return row


def _appeal_view(row: sqlite3.Row) -> dict:
    appeal = dict(row)
    answer_json = appeal.pop("answer_json")
    appeal["answer"] = json.loads(answer_json) if answer_json is not None else None
    return appeal


def _snapshot(conversation: dict, checklist: dict, answers: list[dict]) -> dict:
    return {
        "answers": [
            {
                **answer,
                "evidence_messages": (
                    resolve_evidence(conversation, answer["evidence"]) if answer["evidence"] else []
                ),
            }
            for answer in answers
        ],
        "score": score(checklist, answers),
    }


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: PathID, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN")
        row = _evaluation_row(connection, evaluation_id)
        conversation = get_conversation(connection, row["conversation_id"])
        checklist = get_checklist_version(connection, row["checklist_version_id"])
        appeals = [
            _appeal_view(appeal)
            for appeal in connection.execute(
                "SELECT * FROM appeals WHERE evaluation_id = ? ORDER BY id", (evaluation_id,)
            )
        ]

    original_answers = json.loads(row["answers_json"])
    changes = {
        appeal["criterion_id"]: appeal
        for appeal in appeals
        if appeal["status"] == "closed" and appeal["decision"] == "change"
    }
    final_answers = []
    for answer in original_answers:
        change = changes.get(answer["criterion_id"])
        if change is not None:
            answer = {**answer, "answer": change["answer"], "reason": change["reason"], "evidence": {}}
        final_answers.append(answer)
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "conversation": conversation,
        "checklist": checklist,
        "original": _snapshot(conversation, checklist, original_answers),
        "final": _snapshot(conversation, checklist, final_answers),
        "appeals": appeals,
    }


@router.post("/evaluations/{evaluation_id}/appeals", status_code=201)
def create_appeal(evaluation_id: PathID, payload: AppealRequest, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        evaluation = _evaluation_row(connection, evaluation_id)
        answers = json.loads(evaluation["answers_json"])
        if all(answer["criterion_id"] != payload.criterion_id for answer in answers):
            raise HTTPException(404, "Criterion not found in this evaluation")
        existing = connection.execute(
            "SELECT id FROM appeals WHERE evaluation_id = ? AND criterion_id = ? AND status = 'open'",
            (evaluation_id, payload.criterion_id),
        ).fetchone()
        if existing is not None:
            raise HTTPException(409, "There is already an open appeal for this criterion")
        appeal_id = connection.execute(
            "INSERT INTO appeals (evaluation_id, criterion_id, comment) VALUES (?, ?, ?)",
            (evaluation_id, payload.criterion_id, payload.comment),
        ).lastrowid
        row = connection.execute("SELECT * FROM appeals WHERE id = ?", (appeal_id,)).fetchone()
        return _appeal_view(row)


@router.post("/appeals/{appeal_id}/decision")
def decide_appeal(appeal_id: PathID, payload: DecisionRequest, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        appeal = connection.execute("SELECT * FROM appeals WHERE id = ?", (appeal_id,)).fetchone()
        if appeal is None:
            raise HTTPException(404, "Appeal not found")
        if appeal["status"] != "open":
            raise HTTPException(409, "Appeal is already closed")

        answer_json = None
        if payload.decision == "change":
            evaluation = _evaluation_row(connection, appeal["evaluation_id"])
            checklist = get_checklist_version(connection, evaluation["checklist_version_id"])
            criterion = next(item for item in checklist["criteria"] if item["id"] == appeal["criterion_id"])
            try:
                validate_answer(criterion, payload.answer)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            answer_json = json.dumps(payload.answer)

        connection.execute(
            """
            UPDATE appeals
            SET status = 'closed', decision = ?, reason = ?, answer_json = ?,
                resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (payload.decision, payload.reason, answer_json, appeal_id),
        )
        row = connection.execute("SELECT * FROM appeals WHERE id = ?", (appeal_id,)).fetchone()
        return _appeal_view(row)
