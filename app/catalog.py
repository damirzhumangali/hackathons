import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Request, status
from fastapi import Path as PathParam
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from .db import connect

router = APIRouter(prefix="/api", tags=["Catalog"])


def require_utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("expected valid Unicode text") from None
    return value


UTF8Text = Annotated[str, AfterValidator(require_utf8)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1), AfterValidator(require_utf8)]
GroupID = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256), AfterValidator(require_utf8)
]
PathID = Annotated[int, PathParam(gt=0, le=2**63 - 1)]
Channel = Literal["agent chat", "voice call"]
Language = Literal["RU", "KZ", "EN"]
Speaker = Literal["client", "operator", "bot", "system"]
STARTER_CHECKLIST = Path(__file__).with_name("starter_checklist.json")
ORGANIZER_RESPONSE_LIMIT = 32 * 1024 * 1024


class MessageInput(BaseModel):
    speaker: Speaker
    text: UTF8Text


class ChatMessageInput(MessageInput):
    timestamp: Annotated[str, StringConstraints(pattern=r"^[0-9]{2,}:[0-5][0-9]$")]


class CallMessageInput(MessageInput):
    timestamp: str

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ]", value) is None:
            raise ValueError("expected an ISO 8601 timestamp with a time zone")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("expected an ISO 8601 timestamp with a time zone") from error
        if parsed.utcoffset() is None:
            raise ValueError("expected an ISO 8601 timestamp with a time zone")
        return value


class RecordInput(BaseModel):
    conversation_id: NonEmptyText
    language: Language
    audio_url: UTF8Text | None = None

    @field_validator("audio_url")
    @classmethod
    def validate_audio_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        try:
            url = urlsplit(value)
            port = url.port
            if (
                url.scheme not in {"https", "http"}
                or not url.hostname
                or port == 0
                or "\\" in value
                or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                raise ValueError
        except ValueError:
            raise ValueError("expected an HTTP(S) audio URL with a valid host and port") from None
        return value


class ChatInput(RecordInput):
    channel: Literal["agent chat"]
    transcript: Annotated[list[ChatMessageInput], Field(min_length=1)]


class CallInput(RecordInput):
    channel: Literal["voice call"]
    transcript: Annotated[list[CallMessageInput], Field(min_length=1)]


ConversationInput = Annotated[ChatInput | CallInput, Field(discriminator="channel")]


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: GroupID
    source: Literal["json", "organizers"] | None = None
    records: list[ConversationInput] | None = None

    @field_validator("records")
    @classmethod
    def reject_null_records(cls, records: list[ConversationInput] | None) -> list[ConversationInput]:
        if records is None:
            raise ValueError("records must be an array when provided")
        return records

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "json" and self.records is None:
            raise ValueError("records are required for a JSON import")
        if self.source == "organizers" and self.records is not None:
            raise ValueError("records cannot be provided for an organizer import")
        return self


class OrganizerData(BaseModel):
    dataset_group_id: GroupID
    total_records: Annotated[int, Field(strict=True, ge=0)]
    records: list[ConversationInput]


class ScaleOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
    description: NonEmptyText


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    name: NonEmptyText
    section: NonEmptyText | None = None
    kind: Literal["boolean", "scale", "flag"]
    weight: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    description: NonEmptyText
    na_condition: NonEmptyText | None = None
    options: list[ScaleOption] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scoring(self):
        if self.kind == "flag" and self.weight != 0:
            raise ValueError("a flag must have zero weight")
        if self.kind == "flag" and self.na_condition is not None:
            raise ValueError("a flag must have a yes or no answer")
        if self.kind == "scale":
            values = [option.value for option in self.options]
            if len(values) < 2 or len(set(values)) != len(values) or max(values) == 0:
                raise ValueError("a scale needs at least two distinct steps and a positive maximum")
        elif self.options:
            raise ValueError("scale options are only allowed for scale criteria")
        return self


class ChecklistInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyText
    criteria: Annotated[list[Criterion], Field(min_length=1)]

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, criteria: list[Criterion]) -> list[Criterion]:
        ids = [criterion.id for criterion in criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion IDs must be unique")
        try:
            float(sum(Fraction(str(criterion.weight)) for criterion in criteria))
        except OverflowError:
            raise ValueError("the sum of criterion weights is too large") from None
        return criteria


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversation_versions (
            id INTEGER PRIMARY KEY,
            group_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            channel TEXT NOT NULL,
            language TEXT NOT NULL,
            audio_url TEXT,
            transcript TEXT NOT NULL,
            UNIQUE (group_id, conversation_id, version)
        );

        CREATE TABLE IF NOT EXISTS conversation_heads (
            group_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            version_id INTEGER NOT NULL REFERENCES conversation_versions(id),
            PRIMARY KEY (group_id, conversation_id)
        );

        CREATE TABLE IF NOT EXISTS checklist_versions (
            id INTEGER PRIMARY KEY,
            checklist_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            name TEXT NOT NULL,
            criteria TEXT NOT NULL,
            UNIQUE (checklist_id, version)
        );
        """
    )
    starter = ChecklistInput.model_validate_json(STARTER_CHECKLIST.read_text(encoding="utf-8"))
    with connection:
        connection.execute(
            """
            INSERT INTO conversation_heads (group_id, conversation_id, version_id)
            SELECT group_id, conversation_id, MAX(id)
            FROM conversation_versions
            GROUP BY group_id, conversation_id
            ON CONFLICT (group_id, conversation_id) DO NOTHING
            """
        )
        connection.execute(
            """
            INSERT INTO checklist_versions (checklist_id, version, name, criteria)
            VALUES (1, 1, ?, ?)
            ON CONFLICT (checklist_id, version) DO NOTHING
            """,
            (starter.name, encode_criteria(starter)),
        )


def encode_criteria(checklist: ChecklistInput) -> str:
    return json.dumps(checklist.model_dump(mode="json")["criteria"], ensure_ascii=False)


def conversation_data(row: sqlite3.Row, *, include_transcript: bool = True) -> dict:
    data = dict(row)
    transcript = data.pop("transcript")
    if include_transcript:
        data["transcript"] = json.loads(transcript)
    return data


def get_conversation(connection: sqlite3.Connection, id: int) -> dict | None:
    row = connection.execute("SELECT * FROM conversation_versions WHERE id = ?", (id,)).fetchone()
    return conversation_data(row) if row else None


def list_conversation_ids(connection: sqlite3.Connection, group_id: str) -> list[int]:
    rows = connection.execute(
        """
        SELECT version_id
        FROM conversation_heads
        WHERE group_id = ?
        ORDER BY version_id
        """,
        (group_id,),
    )
    return [row["version_id"] for row in rows]


def checklist_data(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["criteria"] = json.loads(data["criteria"])
    return data


def get_checklist_version(connection: sqlite3.Connection, id: int) -> dict | None:
    row = connection.execute("SELECT * FROM checklist_versions WHERE id = ?", (id,)).fetchone()
    return checklist_data(row) if row else None


def fetch_organizer_records(group_id: str) -> list[ConversationInput]:
    configured_url = os.environ.get("CQA_DATASET_URL", "").strip()
    if not configured_url:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "CQA_DATASET_URL is not configured")
    try:
        url = urlsplit(configured_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("invalid URL")
        query = [(key, value) for key, value in parse_qsl(url.query, keep_blank_values=True) if key != "group_id"]
        query.append(("group_id", group_id))
        request_url = urlunsplit(url._replace(query=urlencode(query), fragment=""))
    except ValueError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Invalid CQA_DATASET_URL") from error

    try:
        request = URLRequest(request_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=25) as response:
            content = response.read(ORGANIZER_RESPONSE_LIMIT + 1)
        if len(content) > ORGANIZER_RESPONSE_LIMIT:
            raise ValueError("response too large")
        payload = json.loads(content)
    except (OSError, ValueError) as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Unable to load organizer dataset") from error
    if not isinstance(payload, dict) or payload.get("isSuccess") is not True:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Organizer API reported an unsuccessful import")
    try:
        dataset = OrganizerData.model_validate(payload.get("data"))
    except ValidationError as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            {
                "message": "Organizer dataset does not match the expected format",
                "errors": error.errors(include_input=False, include_context=False, include_url=False),
            },
        ) from error
    if dataset.dataset_group_id != group_id or dataset.total_records != len(dataset.records):
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Organizer dataset metadata does not match the request")
    return dataset.records


def store_conversations(
    connection: sqlite3.Connection,
    group_id: str,
    records: list[ConversationInput],
) -> dict:
    conversation_ids = []
    imported = 0
    reactivated = 0
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        for record in records:
            messages = [
                {"id": f"m{index}", **message.model_dump()}
                for index, message in enumerate(record.transcript, start=1)
            ]
            transcript = json.dumps(messages, ensure_ascii=False)
            existing = connection.execute(
                """
                SELECT id FROM conversation_versions
                WHERE group_id = ? AND conversation_id = ?
                  AND channel = ? AND language = ? AND audio_url IS ? AND transcript = ?
                ORDER BY version DESC LIMIT 1
                """,
                (group_id, record.conversation_id, record.channel, record.language, record.audio_url, transcript),
            ).fetchone()
            if existing:
                version_id = existing["id"]
            else:
                version_id = connection.execute(
                    """
                    INSERT INTO conversation_versions
                        (group_id, conversation_id, version, channel, language, audio_url, transcript)
                    VALUES (
                        ?, ?,
                        (SELECT COALESCE(MAX(version), 0) + 1
                         FROM conversation_versions WHERE group_id = ? AND conversation_id = ?),
                        ?, ?, ?, ?
                    )
                    RETURNING id
                    """,
                    (
                        group_id, record.conversation_id, group_id, record.conversation_id,
                        record.channel, record.language, record.audio_url, transcript,
                    ),
                ).fetchone()["id"]
                imported += 1
            changed = connection.execute(
                """
                INSERT INTO conversation_heads (group_id, conversation_id, version_id)
                VALUES (?, ?, ?)
                ON CONFLICT (group_id, conversation_id) DO UPDATE
                    SET version_id = excluded.version_id
                    WHERE conversation_heads.version_id != excluded.version_id
                """,
                (group_id, record.conversation_id, version_id),
            ).rowcount
            if existing:
                reactivated += changed
            conversation_ids.append(version_id)
    return {
        "group_id": group_id,
        "total_records": len(records),
        "imported": imported,
        "reactivated": reactivated,
        "unchanged": len(records) - imported - reactivated,
        "conversation_ids": conversation_ids,
    }


@router.post("/imports")
def import_conversations(payload: ImportRequest, request: Request) -> dict:
    records = payload.records
    if records is None:
        records = fetch_organizer_records(payload.group_id)
    seen = set()
    for index, record in enumerate(records):
        if record.conversation_id in seen:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                [{"loc": ["body", "records", index, "conversation_id"], "msg": "duplicate conversation ID"}],
            )
        seen.add(record.conversation_id)
    with closing(connect(request.app.state.database_path)) as connection:
        return store_conversations(connection, payload.group_id, records)


@router.get("/conversations")
def list_conversations(
    request: Request,
    group_id: GroupID | None = None,
    language: Language | None = None,
    channel: Channel | None = None,
) -> list[dict]:
    with closing(connect(request.app.state.database_path)) as connection:
        rows = connection.execute(
            """
            SELECT versions.* FROM conversation_heads AS heads
            JOIN conversation_versions AS versions ON versions.id = heads.version_id
            WHERE (? IS NULL OR heads.group_id = ?)
              AND (? IS NULL OR versions.language = ?)
              AND (? IS NULL OR versions.channel = ?)
            ORDER BY heads.group_id, heads.conversation_id
            """,
            (group_id, group_id, language, language, channel, channel),
        )
        return [conversation_data(row, include_transcript=False) for row in rows]


@router.get("/conversations/{id}")
def read_conversation(id: PathID, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection:
        conversation = get_conversation(connection, id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation version not found")
    return conversation


@router.get("/conversations/{id}/versions")
def list_conversation_versions(id: PathID, request: Request) -> list[dict]:
    with closing(connect(request.app.state.database_path)) as connection:
        rows = connection.execute(
            """
            SELECT versions.* FROM conversation_versions AS versions
            JOIN conversation_versions AS selected
              ON versions.group_id = selected.group_id
             AND versions.conversation_id = selected.conversation_id
            WHERE selected.id = ?
            ORDER BY versions.version DESC
            """,
            (id,),
        )
        versions = [conversation_data(row, include_transcript=False) for row in rows]
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation version not found")
    return versions


@router.get("/checklists")
def list_checklists(request: Request) -> list[dict]:
    with closing(connect(request.app.state.database_path)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM checklist_versions
            WHERE id IN (SELECT MAX(id) FROM checklist_versions GROUP BY checklist_id)
            ORDER BY checklist_id
            """
        )
        return [checklist_data(row) for row in rows]


@router.get("/checklist-versions/{id}")
def read_checklist_version(id: PathID, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection:
        checklist = get_checklist_version(connection, id)
    if checklist is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Checklist version not found")
    return checklist


@router.get("/checklists/{id}/versions")
def list_checklist_versions(id: PathID, request: Request) -> list[dict]:
    with closing(connect(request.app.state.database_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM checklist_versions WHERE checklist_id = ? ORDER BY version DESC",
            (id,),
        )
        versions = [checklist_data(row) for row in rows]
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Checklist not found")
    return versions


@router.post("/checklists/{id}/versions", status_code=status.HTTP_201_CREATED)
def save_checklist_version(id: PathID, payload: ChecklistInput, request: Request) -> dict:
    with closing(connect(request.app.state.database_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        latest = connection.execute(
            "SELECT MAX(version) FROM checklist_versions WHERE checklist_id = ?",
            (id,),
        ).fetchone()[0]
        if latest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Checklist not found")
        row = connection.execute(
            """
            INSERT INTO checklist_versions (checklist_id, version, name, criteria)
            VALUES (?, ?, ?, ?) RETURNING *
            """,
            (id, latest + 1, payload.name, encode_criteria(payload)),
        ).fetchone()
        return checklist_data(row)
