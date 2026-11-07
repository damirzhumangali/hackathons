import copy
import io
import json
import os
import unittest
from email.utils import formatdate
from unittest.mock import patch
from urllib.error import HTTPError

from app.evaluation import EvaluationError, evaluate, resolve_evidence, score


def criterion(criterion_id, weight, *, kind="boolean", section="Качество", na_condition=None, values=()):
    return {
        "id": criterion_id,
        "name": criterion_id,
        "section": section,
        "kind": kind,
        "weight": weight,
        "description": "Правило из выбранной версии",
        "na_condition": na_condition,
        "options": [{"value": value, "description": f"Ступень {value}"} for value in values],
    }


def answer(criterion_id, value):
    return {"criterion_id": criterion_id, "answer": value}


class ScoringTests(unittest.TestCase):
    def test_normalizes_na_preserves_scale_and_reports_flag_separately(self):
        checklist = {"criteria": [
            criterion("consultation", 59.5),
            criterion("dialogue", 15, kind="scale", values=(0, 4, 7, 10)),
            criterion("identity", 20, na_condition="Общий вопрос", section="Безопасность"),
            criterion("closure", 5.5),
            criterion("violation", 0, kind="flag", section="Нарушения"),
        ]}
        answers = [
            answer("consultation", True), answer("dialogue", 7), answer("identity", "N/A"),
            answer("closure", False), answer("violation", True),
        ]

        result = score(checklist, answers)
        self.assertEqual(result["total"], 87.5)
        self.assertEqual(result["applicable_weight"], 80)
        self.assertEqual(result["criteria"][1]["base_points"], 10.5)
        self.assertIsNone(result["criteria"][2]["effective_weight"])
        self.assertEqual(result["gross_flags"], ["violation"])
        self.assertEqual(result["sections"], [
            {"section": "Качество", "total": 87.5, "applicable_weight": 80},
            {"section": "Безопасность", "total": None, "applicable_weight": 0},
        ])
        answers[-1]["answer"] = False
        self.assertEqual(score(checklist, answers)["total"], 87.5)
        answers[2]["answer"] = False
        self.assertEqual(score(checklist, answers)["total"], 70)

    def test_rounds_only_total_and_returns_null_without_applicable_weight(self):
        checklist = {"criteria": [criterion("positive", 1), criterion("negative", 5)]}
        result = score(checklist, [answer("positive", True), answer("negative", False)])
        self.assertEqual(result["total"], 16.67)
        self.assertAlmostEqual(result["criteria"][0]["points"], 100 / 6)
        checklist = {"criteria": [criterion("only", 20, na_condition="Общий вопрос")]}
        self.assertIsNone(score(checklist, [answer("only", "N/A")])["total"])

    def test_half_up_rounding_after_exact_sum_of_scales(self):
        checklist = {"criteria": [
            criterion("contact", 9),
            criterion("consultation", 10, kind="scale", values=(0, 11, 15)),
            criterion("quality", 2, kind="scale", values=(0, 2, 3)),
            criterion("closure", 11, kind="scale", values=(0, 2, 3)),
        ]}
        answers = [answer("contact", True), answer("consultation", 11), answer("quality", 2), answer("closure", 2)]
        # Exact earned weight is 25 out of 32, so 78.125 must round up.
        result = score(checklist, answers)
        self.assertEqual(result["total"], 78.13)
        self.assertEqual(result["sections"][0]["total"], 78.13)

    def test_legacy_overweight_checklist_remains_json_serializable(self):
        checklist = {"criteria": [criterion("positive", 1e308), criterion("negative", 1e308)]}
        result = score(checklist, [answer("positive", True), answer("negative", False)])
        self.assertEqual(result["total"], 50)
        self.assertIsNone(result["applicable_weight"])
        self.assertIsNone(result["sections"][0]["applicable_weight"])
        json.dumps(result, allow_nan=False)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.conversation = {
            "language": "KZ", "channel": "voice call",
            "transcript": [
                {"id": "m1", "timestamp": "2025-01-01T09:00:00+05:00", "speaker": "client", "text": "Сәлеметсіз бе"},
                {"id": "m2", "timestamp": "2025-01-01T09:00:05+05:00", "speaker": "operator", "text": "Сәлеметсіз бе"},
                {"id": "m3", "timestamp": "2025-01-01T09:00:10+05:00", "speaker": "system", "text": "Call ended"},
            ],
        }
        self.checklist = {"version": 2, "criteria": [
            criterion("new-rule", 10),
            criterion("dialogue", 15, kind="scale", values=(0, 4, 7, 10)),
        ]}
        self.answers = [
            {"criterion_id": "new-rule", "answer": False, "reason": "Действие пропущено.",
             "evidence": {"start_id": "m2", "end_id": "m3"}, "omission": True},
            {"criterion_id": "dialogue", "answer": 7, "reason": "Уточнил ответ.",
             "evidence": {"message_ids": ["m2"]}, "omission": False},
        ]
        config = patch.dict(os.environ, {
            "CQA_LLM_URL": "http://127.0.0.1:11434/v1/chat/completions",
            "CQA_LLM_MODEL": "test-model",
            "CQA_LLM_TIMEOUT": "4.5",
        }, clear=True)
        config.start()
        self.addCleanup(config.stop)

    def mock_completion(self, opener, answers):
        payload = {"choices": [{"message": {"content": json.dumps({"answers": answers})}}]}
        opener.return_value.open.return_value = io.BytesIO(json.dumps(payload).encode())

    @patch("app.evaluation.build_opener")
    def test_calls_model_with_current_rubric_and_resolves_original_messages(self, opener):
        self.mock_completion(opener, self.answers)
        result = evaluate(self.conversation, self.checklist)
        self.assertEqual(result, [
            {key: value for key, value in item.items() if key != "omission"} for item in self.answers
        ])
        request = opener.return_value.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", body)
        self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 4.5)
        self.assertIsNone(request.get_header("Authorization"))
        sent = json.loads(body["messages"][1]["content"])
        self.assertEqual(sent["criteria"], self.checklist["criteria"])
        self.assertEqual(sent["transcript"], self.conversation["transcript"])
        self.assertEqual(
            resolve_evidence(self.conversation, result[0]["evidence"]), self.conversation["transcript"][1:]
        )

    @patch("app.evaluation.build_opener")
    def test_strict_mode_constrains_current_criteria_answers_and_evidence(self, opener):
        self.checklist["criteria"][1]["na_condition"] = "Диалог не состоялся."
        answers = {item["criterion_id"]: item for item in copy.deepcopy(self.answers)}
        answers["dialogue"]["evidence"] = {"start_id": "m2", "end_id": "m2"}
        # Provider object order is irrelevant to the order of the public list.
        self.mock_completion(opener, dict(reversed(list(answers.items()))))
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            result = evaluate(self.conversation, self.checklist)

        self.assertEqual([item["criterion_id"] for item in result], ["new-rule", "dialogue"])
        self.assertEqual(result[1]["answer"], 7)
        self.assertEqual(result[1]["evidence"], {"start_id": "m2", "end_id": "m2"})
        self.assertNotIn("omission", result[1])
        body = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertEqual(body["temperature"], 0)
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        keyed = response_format["json_schema"]["schema"]["properties"]["answers"]
        self.assertEqual(keyed["required"], ["new-rule", "dialogue"])
        self.assertFalse(keyed["additionalProperties"])
        fields = keyed["properties"]["dialogue"]["properties"]
        self.assertEqual(fields["answer"]["enum"], [0, 4, 7, 10, "N/A"])
        self.assertEqual(fields["criterion_id"]["enum"], ["dialogue"])
        self.assertEqual(fields["evidence"]["required"], ["start_id", "end_id"])
        self.assertEqual(fields["evidence"]["properties"]["end_id"]["enum"], ["m1", "m2", "m3"])

    @patch("app.evaluation.build_opener")
    def test_strict_mode_validates_object_and_range_even_if_provider_ignores_schema(self, opener):
        answers = {item["criterion_id"]: item for item in copy.deepcopy(self.answers)}
        answers["dialogue"]["evidence"] = {"start_id": "m2", "end_id": "m2"}
        mutations = [
            (lambda result: result.pop("dialogue"), "Ключи ответов"),
            (lambda result: result["dialogue"].update(criterion_id="new-rule"), "criterion_id"),
            (lambda result: result["dialogue"].update(extra=True), "не соответствуют схеме"),
            (lambda result: result["dialogue"].update(evidence={"message_ids": ["m2"]}), "границы"),
            (lambda result: result["dialogue"].update(evidence={"start_id": "m0", "end_id": "m2"}), "репликами"),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for mutate, message in mutations:
                with self.subTest(message=message):
                    mutated = copy.deepcopy(answers)
                    mutate(mutated)
                    self.mock_completion(opener, mutated)
                    with self.assertRaisesRegex(EvaluationError, message):
                        evaluate(self.conversation, self.checklist)

            content = '{"answers":' + json.dumps(answers) + ',"answers":' + json.dumps(answers) + '}'
            payload = {"choices": [{"message": {"content": content}}]}
            opener.return_value.open.return_value = io.BytesIO(json.dumps(payload).encode())
            with self.assertRaisesRegex(EvaluationError, "повторяющийся ключ"):
                evaluate(self.conversation, self.checklist)

    @patch("app.evaluation.build_opener")
    def test_rejects_incomplete_invalid_or_ungrounded_answers(self, opener):
        mutations = [
            (lambda answers: answers.pop(), "не по всем"),
            (lambda answers: answers.append(answers[0]), "несколько раз"),
            (lambda answers: answers[1].update(answer=6), "допустимую шкалу"),
            (lambda answers: answers[1].update(answer=True), "допустимую шкалу"),
            (lambda answers: answers[0].update(answer="N/A"), "не разрешён"),
            (lambda answers: answers[0].update(reason=" "), "нет обоснования"),
            (lambda answers: answers[0].update(reason="\ud800"), "Unicode"),
            (lambda answers: answers[0].update(evidence={"message_ids": ["missing"]}), "неизвестные реплики"),
            (lambda answers: answers[0].update(evidence={"start_id": "m3", "end_id": "m2"}), "предшествовать"),
            (lambda answers: answers[0].update(omission=False, evidence={"message_ids": ["m3"]}), "границы"),
            (lambda answers: answers[1].update(omission=True), "границы"),
            (lambda answers: answers[1].pop("omission"), "должен быть логическим"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                answers = copy.deepcopy(self.answers)
                mutate(answers)
                self.mock_completion(opener, answers)
                with self.assertRaisesRegex(EvaluationError, message):
                    evaluate(self.conversation, self.checklist)

    @patch("app.evaluation.build_opener")
    def test_distinguishes_temporary_failures_from_invalid_credentials(self, opener):
        for status, retryable in ((429, True), (503, True), (401, False)):
            with self.subTest(status=status):
                opener.return_value.open.side_effect = HTTPError("http://mock", status, "error", {}, io.BytesIO())
                with self.assertRaisesRegex(EvaluationError, f"HTTP {status}") as context:
                    evaluate(self.conversation, self.checklist)
                self.assertEqual(context.exception.retryable, retryable)
        opener.return_value.open.side_effect = TimeoutError()
        with self.assertRaisesRegex(EvaluationError, "вовремя") as context:
            evaluate(self.conversation, self.checklist)
        self.assertTrue(context.exception.retryable)

    @patch("app.evaluation.build_opener")
    def test_retry_after_supports_seconds_date_and_bound(self, opener):
        now = 1_700_000_000
        cases = [
            (429, "20", 20),
            (503, formatdate(now + 35, usegmt=True), 35),
            (503, formatdate(now - 10, usegmt=True), 0),
            (429, "7200", 300),
            (429, "invalid", None),
            (401, "20", None),
        ]
        with patch("app.evaluation.time.time", return_value=now):
            for status, header, expected in cases:
                with self.subTest(status=status, header=header):
                    opener.return_value.open.side_effect = HTTPError(
                        "http://mock", status, "error", {"Retry-After": header}, io.BytesIO()
                    )
                    with self.assertRaises(EvaluationError) as context:
                        evaluate(self.conversation, self.checklist)
                    self.assertEqual(context.exception.retry_after, expected)

    @patch("app.evaluation.build_opener")
    def test_slow_body_cannot_extend_request_deadline(self, opener):
        clock = 0.0

        class SlowBody(io.BytesIO):
            def read1(self, size):
                nonlocal clock
                clock += 2
                return super().read1(min(size, 5))

        payload = {"choices": [{"message": {"content": json.dumps({"answers": self.answers})}}]}
        opener.return_value.open.return_value = SlowBody(json.dumps(payload).encode())
        with patch("app.evaluation.time.monotonic", side_effect=lambda: clock):
            with self.assertRaisesRegex(EvaluationError, "вовремя") as context:
                evaluate(self.conversation, self.checklist)
        self.assertTrue(context.exception.retryable)
        self.assertEqual(clock, 6)
