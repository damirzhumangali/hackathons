import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import patch

from app.db import connect, initialize_database
from app.main import app


class StartupTests(unittest.TestCase):
    def test_concurrent_initialization_of_new_database(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=4) as pool:
            for index in range(8):
                path = Path(directory) / f"cqa-{index}.sqlite3"
                barrier = Barrier(4)

                def initialize():
                    barrier.wait(timeout=5)
                    initialize_database(path)

                tasks = [pool.submit(initialize) for _ in range(4)]
                for task in tasks:
                    task.result(timeout=10)
                with closing(connect(path)) as connection:
                    self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM checklist_versions").fetchone()[0], 1)
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluation_jobs").fetchone()[0], 0)


class ApiValidationTests(unittest.IsolatedAsyncioTestCase):
    async def post_raw_json(self, path, body):
        response = []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            response.append(message)

        await app({
            "type": "http", "method": "POST", "scheme": "http", "path": path,
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
        }, receive, send)
        return response[0]["status"], json.loads(response[1]["body"])

    async def test_validation_returns_422_for_input_that_cannot_be_echoed_as_json(self):
        cases = [
            (b'{"checklist_version_id":1e309,"conversation_ids":[1]}', ["body", "checklist_version_id"]),
            (br'{"checklist_version_id":1,"conversation_ids":[1],"\ud800":0}', ["body"]),
        ]
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"CQA_DB_PATH": f"{directory}/api.sqlite3"}):
            async with app.router.lifespan_context(app):
                for body, location in cases:
                    with self.subTest(location=location):
                        status, response = await self.post_raw_json("/api/evaluation-runs", body)
                        self.assertEqual(status, 422)
                        self.assertEqual(response["detail"][0]["loc"], location)
                        self.assertNotIn("input", response["detail"][0])
