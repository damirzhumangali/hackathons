import json
import os
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.db import connect, initialize_database
from app.main import app


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.path = Path(directory) / "catalog.sqlite3"
        self.enterContext(patch.dict(os.environ, {"CQA_DB_PATH": str(self.path)}))
        await self.enterAsyncContext(app.router.lifespan_context(app))

    async def request(self, method, path, payload=None):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        messages = []

        async def receive():
            return {"type": "http.request", "body": body}

        async def send(message):
            messages.append(message)

        await app(
            {
                "type": "http",
                "method": method,
                "path": path,
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "scheme": "http",
                "root_path": "",
                "http_version": "1.1",
            },
            receive,
            send,
        )
        response = b"".join(message["body"] for message in messages if message["type"] == "http.response.body")
        return messages[0]["status"], json.loads(response)

    @staticmethod
    def record(text="Привет", *, channel="agent chat", timestamp="00:01", audio_url=None):
        return {
            "conversation_id": "same",
            "channel": channel,
            "language": "RU",
            "transcript": [{"timestamp": timestamp, "speaker": "operator", "text": text}],
            "audio_url": audio_url,
        }

    async def test_reimport_activates_existing_snapshot_for_list_and_batch(self):
        async def upload(text):
            status, result = await self.request(
                "POST", "/api/imports", {"group_id": "DEMO", "records": [self.record(text)]}
            )
            self.assertEqual(status, 200)
            return result

        first = await upload("A")
        second = await upload("B")
        a = first["conversation_ids"][0]
        b = second["conversation_ids"][0]
        self.assertNotEqual(a, b)

        # Simulate an existing database with versions created before active heads were introduced.
        with closing(connect(self.path)) as connection:
            connection.execute("DROP TABLE conversation_heads")
        initialize_database(self.path)
        self.assertEqual([row["id"] for row in (await self.request("GET", "/api/conversations"))[1]], [b])

        restored = await upload("A")
        self.assertEqual((restored["imported"], restored["reactivated"], restored["unchanged"]), (0, 1, 0))
        self.assertEqual(restored["conversation_ids"], [a])
        duplicate = await upload("A")
        self.assertEqual((duplicate["imported"], duplicate["reactivated"], duplicate["unchanged"]), (0, 0, 1))

        initialize_database(self.path)
        self.assertEqual([row["id"] for row in (await self.request("GET", "/api/conversations"))[1]], [a])
        self.assertEqual(
            [row["id"] for row in (await self.request("GET", f"/api/conversations/{a}/versions"))[1]], [b, a]
        )
        self.assertEqual((await self.request("GET", f"/api/conversations/{b}"))[1]["transcript"][0]["text"], "B")
        status, batch = await self.request(
            "POST", "/api/evaluation-runs", {"group_id": "DEMO", "checklist_version_id": 1}
        )
        self.assertEqual(status, 202)
        self.assertEqual([job["conversation_id"] for job in batch["jobs"]], [a])

    async def test_malformed_imports_return_field_errors(self):
        for payload, field in [
            ({"group_id": "DEMO", "records": None}, "records"),
            ({"group_id": "x" * 257, "records": []}, "group_id"),
            ({"group_id": "DEMO", "records": [self.record("\ud800")]}, "text"),
            (
                {"group_id": "DEMO", "records": [self.record(
                    channel="voice call", timestamp="2026-11-07x12:01:02+05:00"
                )]},
                "timestamp",
            ),
            (
                {"group_id": "DEMO", "records": [self.record(
                    audio_url="https://example.test:abcd/call"
                )]},
                "audio_url",
            ),
        ]:
            with self.subTest(field=field):
                status, response = await self.request("POST", "/api/imports", payload)
                self.assertEqual(status, 422)
                self.assertEqual(response["detail"][0]["loc"][-1], field)

        timestamp = "2026-11-07T12:01:02+05:00"
        status, imported = await self.request("POST", "/api/imports", {
            "group_id": "DEMO", "records": [self.record(channel="voice call", timestamp=timestamp)]
        })
        self.assertEqual(status, 200)
        _, conversation = await self.request("GET", f"/api/conversations/{imported['conversation_ids'][0]}")
        self.assertEqual(conversation["transcript"][0]["timestamp"], timestamp)
        self.assertIsNone(conversation["audio_url"])

    async def test_catalog_rejects_oversized_numeric_inputs(self):
        sqlite_overflow = 2**63
        for path in [
            f"/api/conversations/{sqlite_overflow}",
            f"/api/conversations/{sqlite_overflow}/versions",
            f"/api/checklist-versions/{sqlite_overflow}",
            f"/api/checklists/{sqlite_overflow}/versions",
        ]:
            with self.subTest(path=path):
                status, response = await self.request("GET", path)
                self.assertEqual(status, 422)
                self.assertEqual(response["detail"][0]["loc"], ["path", "id"])

        criterion = {"id": "a", "name": "Правило", "kind": "boolean", "weight": 1e308, "description": "Правило"}
        status, response = await self.request("POST", "/api/checklists/1/versions", {
            "name": "Чек-лист", "criteria": [criterion, {**criterion, "id": "b"}]
        })
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"][0]["loc"], ["body", "criteria"])

        nonfinite = json.dumps({"name": "Чек-лист", "criteria": [criterion]}).replace("1e+308", "1e309")
        status, response = await self.request("POST", "/api/checklists/1/versions", nonfinite.encode())
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"][0]["loc"][-1], "weight")

        scale = {**criterion, "kind": "scale", "weight": 10, "options": [
            {"value": 0, "description": "Ноль"},
            {"value": 2**53, "description": "Максимум"},
        ]}
        status, response = await self.request("POST", "/api/checklists/1/versions", {
            "name": "Чек-лист", "criteria": [scale]
        })
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"][0]["loc"][-1], "value")
