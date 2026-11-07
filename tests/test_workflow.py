import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request
from pydantic import ValidationError

from app import catalog, worker, workflow
from app.db import connect, initialize_database
from app.evaluation import EvaluationError


def answer_checklist(conversation, checklist):
    answers = []
    for criterion in checklist["criteria"]:
        if criterion["kind"] == "scale":
            answer = max(option["value"] for option in criterion["options"])
        else:
            answer = criterion["id"] not in {"03", "08"}
        answers.append(
            {
                "criterion_id": criterion["id"],
                "answer": answer,
                "reason": "Исходное объяснение",
                "evidence": {"message_ids": [conversation["transcript"][0]["id"]]},
            }
        )
    return answers


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.path = Path(directory) / "workflow.sqlite3"
        initialize_database(self.path)
        app = SimpleNamespace(state=SimpleNamespace(database_path=self.path))
        self.request = Request({"type": "http", "app": app})

    def import_chats(self, *conversation_ids):
        records = [
            {
                "conversation_id": conversation_id,
                "channel": "agent chat",
                "language": "RU",
                "transcript": [{"timestamp": "00:01", "speaker": "operator", "text": "Здравствуйте!"}],
            }
            for conversation_id in conversation_ids
        ]
        payload = catalog.ImportRequest(group_id="WORKFLOW_TEST", records=records)
        with closing(connect(self.path)) as connection:
            return catalog.store_conversations(connection, payload.group_id, payload.records)["conversation_ids"]

    def create_batch(self, *conversation_ids):
        ids = self.import_chats(*conversation_ids)
        return workflow.create_run(workflow.RunRequest(checklist_version_id=1, conversation_ids=ids), self.request)

    def test_worker_resumes_after_lock_on_claim(self):
        run = self.create_batch("busy-claim")
        with closing(connect(self.path)) as writer, patch.object(
            worker, "connect", side_effect=self.connect_with_short_timeout
        ), patch.object(worker, "evaluate", side_effect=answer_checklist) as evaluator:
            writer.execute("BEGIN IMMEDIATE")
            self.assertFalse(worker.run_once(self.path))
            self.assertFalse(evaluator.called)
            writer.rollback()
            self.assertTrue(worker.run_once(self.path))

        self.assertEqual(workflow.get_run(run["id"], self.request)["status"], "completed")

    def test_worker_retries_locked_save_without_calling_model_again(self):
        run = self.create_batch("busy-save")
        with closing(connect(self.path)) as writer:
            def evaluate(conversation, checklist):
                writer.execute("BEGIN IMMEDIATE")
                return answer_checklist(conversation, checklist)

            with patch.object(worker, "connect", side_effect=self.connect_with_short_timeout), patch.object(
                worker, "evaluate", side_effect=evaluate
            ) as evaluator, patch.object(worker.time, "sleep", side_effect=lambda _: writer.rollback()) as sleeper:
                with self.assertLogs(worker.logger, level="WARNING"):
                    self.assertTrue(worker.run_once(self.path))
                self.assertEqual(evaluator.call_count, 1)
                self.assertEqual(sleeper.call_count, 1)

        job = workflow.get_run(run["id"], self.request)["jobs"][0]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["attempts"], 1)
        self.assertIsNotNone(job["evaluation_id"])

    @staticmethod
    def connect_with_short_timeout(path):
        connection = connect(path)
        connection.execute("PRAGMA busy_timeout = 1")
        return connection

    def test_unsavable_result_fails_one_job_and_worker_keeps_processing(self):
        run = self.create_batch("bad-result", "good-result")

        def evaluate(conversation, checklist):
            answers = answer_checklist(conversation, checklist)
            if conversation["conversation_id"] == "bad-result":
                answers[0]["reason"] = "\ud800"
            return answers

        with patch.object(worker, "evaluate", side_effect=evaluate), self.assertLogs(worker.logger, level="ERROR"):
            self.assertTrue(worker.run_once(self.path))
            self.assertTrue(worker.run_once(self.path))

        result = workflow.get_run(run["id"], self.request)
        self.assertEqual(result["status"], "completed_with_errors")
        bad, good = result["jobs"]
        self.assertEqual((bad["status"], bad["evaluation_id"]), ("failed", None))
        self.assertEqual(bad["error"], "Could not save evaluation result")
        self.assertEqual(good["status"], "completed")

    def test_provider_retry_after_delays_next_attempt(self):
        run = self.create_batch("rate-limited")
        with patch.object(worker.time, "time", return_value=100) as clock:
            temporary_error = EvaluationError("rate limited", retryable=True, retry_after=20)
            with patch.object(worker, "evaluate", side_effect=temporary_error):
                self.assertTrue(worker.run_once(self.path))
            clock.return_value = 119
            with patch.object(worker, "evaluate", side_effect=answer_checklist) as evaluator:
                self.assertFalse(worker.run_once(self.path))
                self.assertFalse(evaluator.called)
                clock.return_value = 120
                self.assertTrue(worker.run_once(self.path))
                self.assertEqual(evaluator.call_count, 1)

        result = workflow.get_run(run["id"], self.request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["jobs"][0]["attempts"], 2)

    def test_appeal_payload_rejects_non_utf8_text(self):
        with self.assertRaises(ValidationError):
            workflow.AppealRequest(criterion_id="\ud800", comment="Вопрос")
        with self.assertRaises(ValidationError):
            workflow.AppealRequest(criterion_id="03", comment="\ud800")
        with self.assertRaises(ValidationError):
            workflow.DecisionRequest(decision="confirm", reason="\ud800")

    def test_mixed_run_recovers_leased_job_and_keeps_one_result_per_job(self):
        self.import_chats("retry", "restart", "bad")
        run = workflow.create_run(workflow.RunRequest(checklist_version_id=1, group_id="WORKFLOW_TEST"), self.request)
        calls = []

        def evaluate(conversation, checklist):
            name = conversation["conversation_id"]
            calls.append(name)
            if name == "retry" and calls.count(name) == 1:
                raise EvaluationError("Temporary failure", retryable=True)
            if name == "bad":
                raise EvaluationError("Invalid model response")
            return answer_checklist(conversation, checklist)

        with patch.object(worker, "evaluate", side_effect=evaluate), patch.object(worker.time, "time", return_value=100) as clock:
            self.assertTrue(worker.run_once(self.path))
            abandoned = worker._claim_next(self.path, lease_seconds=1)
            self.assertEqual(abandoned.id, run["jobs"][1]["id"])
            clock.return_value = 101.5
            self.assertTrue(worker.run_once(self.path))
            worker._complete(self.path, abandoned, [])
            self.assertTrue(worker.run_once(self.path))
            self.assertFalse(worker.run_once(self.path))
            clock.return_value = 102
            self.assertTrue(worker.run_once(self.path))
            self.assertFalse(worker.run_once(self.path))

        result = workflow.get_run(run["id"], self.request)
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["counts"], {"queued": 0, "running": 0, "completed": 2, "failed": 1, "total": 3})
        self.assertEqual([job["attempts"] for job in result["jobs"]], [2, 2, 1])
        self.assertEqual(calls, ["retry", "restart", "bad", "retry"])
        for job in result["jobs"][:2]:
            evaluation = workflow.get_evaluation(job["evaluation_id"], self.request)
            self.assertEqual(evaluation["original"]["score"]["total"], 80)

    def test_appeal_uses_original_checklist_and_preserves_original_score(self):
        conversation_ids = self.import_chats("appeal")
        payload = workflow.RunRequest(checklist_version_id=1, conversation_ids=conversation_ids)
        run = workflow.create_run(payload, self.request)
        with patch.object(worker, "evaluate", side_effect=answer_checklist):
            self.assertTrue(worker.run_once(self.path))
        evaluation_id = workflow.get_run(run["id"], self.request)["jobs"][0]["evaluation_id"]
        initial = workflow.get_evaluation(evaluation_id, self.request)
        old_checklist = initial["checklist"]

        # The appeal must still work after the disputed criterion has been removed from v2.
        new_criteria = [criterion for criterion in old_checklist["criteria"] if criterion["id"] != "03"]
        version = catalog.save_checklist_version(
            old_checklist["checklist_id"],
            catalog.ChecklistInput(name="Новая версия", criteria=new_criteria),
            self.request,
        )
        self.assertEqual(version["version"], 2)

        appeal = workflow.create_appeal(
            evaluation_id, workflow.AppealRequest(criterion_id="03", comment="Общий вопрос"), self.request
        )
        invalid = workflow.DecisionRequest(decision="change", answer=1, reason="Проверка шкалы")
        with self.assertRaises(HTTPException) as rejected:
            workflow.decide_appeal(appeal["id"], invalid, self.request)
        self.assertEqual(rejected.exception.status_code, 422)

        corrected = workflow.DecisionRequest(decision="change", answer="N/A", reason="Идентификация не нужна")
        workflow.decide_appeal(appeal["id"], corrected, self.request)
        with self.assertRaises(HTTPException) as duplicate:
            workflow.decide_appeal(appeal["id"], corrected, self.request)
        self.assertEqual(duplicate.exception.status_code, 409)

        evaluation = workflow.get_evaluation(evaluation_id, self.request)
        self.assertEqual(evaluation["checklist"]["version"], 1)
        self.assertEqual(evaluation["original"], initial["original"])
        self.assertEqual(evaluation["final"]["score"]["total"], 100)
        self.assertEqual(evaluation["appeals"][0]["status"], "closed")
        self.assertEqual(evaluation["appeals"][0]["answer"], "N/A")
