import copy
import io
import json
import os
import unittest
from email.utils import formatdate
from pathlib import Path
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


def starter_identity_checklist():
    source = Path(__file__).resolve().parents[1] / "app" / "starter_checklist.json"
    criteria = json.loads(source.read_text(encoding="utf-8"))["criteria"]
    return {"criteria": [item for item in criteria if item["id"] == "03"]}


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

    def strict_answers(self):
        answers = {
            item["criterion_id"]: {
                field: value for field, value in copy.deepcopy(item).items() if field not in {"criterion_id", "omission"}
            }
            for item in self.answers
        }
        answers["dialogue"]["evidence"] = {"start_id": "m2", "end_id": "m2"}
        return answers

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
        answers = self.strict_answers()
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
        self.assertEqual(set(fields), {"answer", "reason", "evidence"})
        self.assertEqual(fields["answer"]["enum"], [0, 4, 7, 10, "N/A"])
        self.assertEqual(fields["evidence"]["required"], ["start_id", "end_id"])
        self.assertEqual(fields["evidence"]["properties"]["end_id"]["enum"], ["m1", "m2", "m3"])
        sent = json.loads(body["messages"][1]["content"])
        self.assertEqual(list(sent)[0], "criteria")
        self.assertEqual(sent["transcript"][0], ["m1", "client", "2025-01-01T09:00:00+05:00", "Сәлеметсіз бе"])

    @patch("app.evaluation.build_opener")
    def test_strict_mode_validates_object_and_range_even_if_provider_ignores_schema(self, opener):
        answers = self.strict_answers()
        mutations = [
            (lambda result: result.pop("dialogue"), "Ключи ответов"),
            (lambda result: result["dialogue"].update(criterion_id="new-rule"), "не соответствуют схеме"),
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
    def test_starter_identity_uses_scope_explicit_verification_and_chronology(self, opener):
        checklist = starter_identity_checklist()
        confirmations = [
            ("RU", "Личность клиента подтверждена."),
            ("KZ", "Клиенттің жеке басы расталды."),
            ("EN", "Customer identity verified."),
        ]
        cases = [
            (language, "account", "none", None, "m4", "system", text, True)
            for language, text in confirmations
        ] + [
            ("EN", "public", "consent", "m2", None, "bot", "By continuing, you agree to personal data processing.", "N/A"),
            ("EN", "account", "confirmed", "m2", "m4", "bot", "By continuing, you agree to personal data processing.", False),
            ("EN", "account", "confirmed", "m2", "m4", "client", "My identity is verified.", False),
            ("EN", "account", "confirmed", "m2", "m4", "operator", "Do we need your identity verified?", False),
            ("EN", "account", "confirmed", "m5", "m4", "operator", "We can skip the verification.", False),
            ("RU", "account", "confirmed", "m2", "m4", "system", "Личность клиента подтверждена? Проверьте статус.", False),
            ("RU", "account", "confirmed", "m2", "m4", "system", "Личность клиента подтверждена не была.", False),
            ("EN", "account", "confirmed", "m2", "m4", "system", "Customer identity verification completed unsuccessfully.", False),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for language, scope, verification, verification_id, protected_id, speaker, detail, expected in cases:
                with self.subTest(language=language, scope=scope, detail=detail):
                    conversation = {"language": language, "channel": "agent chat", "transcript": [
                        {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": "Question."},
                        {"id": "m2", "timestamp": "00:02", "speaker": speaker, "text": detail},
                        {"id": "m3", "timestamp": "00:03", "speaker": "client", "text": "Thank you."},
                        {"id": "m4", "timestamp": "00:04", "speaker": "operator", "text": (
                            "The account has a restriction." if scope == "account" else "Open the pricing menu."
                        )},
                        {"id": "m5", "timestamp": "00:05", "speaker": "system", "text": (
                            "Customer identity verified." if scope == "account" else "Chat closed."
                        )},
                    ]}
                    self.mock_completion(opener, {"03": {
                        "scope": scope, "verification": verification, "verification_id": verification_id,
                        "protected_id": protected_id, "evidence": {"start_id": "m1", "end_id": "m3"},
                    }})
                    [result] = evaluate(conversation, checklist)
                    self.assertEqual(result["answer"], expected)
                    self.assertEqual(result["criterion_id"], "03")
                    if expected is True:
                        self.assertEqual(result["evidence"], {"start_id": "m2", "end_id": "m4"})
                    elif expected is False:
                        self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m4"})

        body = json.loads(opener.return_value.open.call_args.args[0].data)
        fields = body["response_format"]["json_schema"]["schema"]["properties"]["answers"]["properties"]["03"]["properties"]
        self.assertEqual(fields["verification_id"]["enum"], ["m1", "m2", "m3", "m4", "m5", None])
        self.assertNotIn("answer", fields)

    @patch("app.evaluation.build_opener")
    def test_deferred_specific_registration_reset_requires_identity(self, opener):
        checklist = starter_identity_checklist()
        cases = [
            ("RU", "Регистрируем его брокерский счёт. Мы зарегистрировались, но вы сможете сбросить нам эти данные?",
             "Назовите номер телефона, с которого начали регистрацию.", False),
            ("RU", "Сбросьте регистрацию моего счёта.",
             "Запрос по регистрации сможем проверить завтра.", False),
            ("EN", "Could you please reset my account registration?",
             "We can review that account request next week.", False),
            ("KZ", "Тіркеуімді жойып бере аласыз ба?",
             "Тіркеу туралы өтінішіңізді ертең қараймыз.", False),
            ("RU", "Можете подсказать, как мне сбросить регистрацию моего счёта?",
             "Откройте настройки регистрации.", "N/A"),
            ("EN", "How do I reset my account registration myself?",
             "Open Registration in the app.", "N/A"),
            ("EN", "Could you delete this message? My account is fine.",
             "I can't remove chat messages.", "N/A"),
            ("EN", "Could you delete this message from my account inbox?",
             "I can't remove chat messages.", "N/A"),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for language, request, reply, expected in cases:
                with self.subTest(language=language, request=request):
                    conversation = {"language": language, "channel": "voice call", "transcript": [
                        {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": request},
                        {"id": "m2", "timestamp": "00:02", "speaker": "operator", "text": reply},
                        {"id": "m3", "timestamp": "00:03", "speaker": "operator", "text": "Goodbye."},
                    ]}
                    self.mock_completion(opener, {"03": {
                        "scope": "public", "verification": "none", "verification_id": None,
                        "protected_id": None, "evidence": {"start_id": "m3", "end_id": "m3"},
                    }})
                    [result] = evaluate(conversation, checklist)
                    self.assertEqual(result["answer"], expected)
                    if expected is False:
                        self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m2"})

    @patch("app.evaluation.build_opener")
    def test_actual_account_restriction_overrides_public_facts_but_conditionals_do_not(self, opener):
        checklist = starter_identity_checklist()
        cases = [
            ("RU", "operator", "Я посмотрел: на вашем счёте есть ограничение на вывод.", False),
            ("KZ", "operator", "Сіздің шотыңызда ақша шығаруға шектеу қойылғанын көріп тұрмын.", False),
            ("EN", "operator", "Hi, I'm Tyler. I can see a withdrawal restriction on your account.", False),
            ("RU", "operator", "Если на вашем счёте есть ограничение на вывод, проверьте уведомления.", "N/A"),
            ("KZ", "operator", "Сіздің шотыңызда ақша шығаруға шектеу қойылған болса, хабарламаны қараңыз.", "N/A"),
            ("EN", "operator", "Is there a withdrawal restriction on your account? Check the Settings menu.", "N/A"),
            ("EN", "operator", "You said there is a withdrawal restriction on your account.", "N/A"),
            ("EN", "client", "There is a withdrawal restriction on your account.", "N/A"),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for language, speaker, text, expected in cases:
                with self.subTest(language=language, text=text):
                    conversation = {"language": language, "channel": "agent chat", "transcript": [
                        {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": "Where are account settings?"},
                        {"id": "m2", "timestamp": "00:02", "speaker": speaker, "text": text},
                    ]}
                    self.mock_completion(opener, {"03": {
                        "scope": "public", "verification": "none", "verification_id": None,
                        "protected_id": None, "evidence": {"start_id": "m1", "end_id": "m1"},
                    }})
                    [result] = evaluate(conversation, checklist)
                    self.assertEqual(result["answer"], expected)
                    if expected is False:
                        self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m2"})

    @patch("app.evaluation.build_opener")
    def test_private_paid_offer_does_not_make_generic_instrument_navigation_protected(self, opener):
        checklist = starter_identity_checklist()
        conversation = {"language": "EN", "channel": "agent chat", "transcript": [
            {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": "How do I find bonds in the app?"},
            {"id": "m2", "timestamp": "00:02", "speaker": "operator", "text": "Open Quotes and search by ticker."},
            {"id": "m3", "timestamp": "00:03", "speaker": "operator", "text": (
                "Send me your personal number and I'll call you after work. I charge for private advice."
            )},
        ]}
        facts = {
            "scope": "account", "verification": "none", "verification_id": None,
            "protected_id": "m3", "evidence": {"start_id": "m3", "end_id": "m3"},
        }
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for protected_id in ("m3", None):
                self.mock_completion(opener, {"03": {**facts, "protected_id": protected_id}})
                [result] = evaluate(conversation, checklist)
                self.assertEqual(result["answer"], "N/A")
                self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m1"})

            self.mock_completion(opener, {"03": {**facts, "scope": "public", "protected_id": "m2"}})
            [result] = evaluate(conversation, checklist)
            self.assertEqual(result["answer"], "N/A")
            self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m1"})

            conversation["transcript"].append({
                "id": "m4", "timestamp": "00:04", "speaker": "client",
                "text": "Could you also check why my transfer is missing?",
            })
            self.mock_completion(opener, {"03": facts})
            [result] = evaluate(conversation, checklist)
            self.assertIs(result["answer"], False)

    @patch("app.evaluation.build_opener")
    def test_individual_handoff_uses_operator_action_after_verification(self, opener):
        checklist = starter_identity_checklist()
        cases = [
            ("RU", "Я перевёл деньги, но они не поступили.", "Личность клиента подтверждена.",
             "Я передал обращение по вашему переводу специалистам."),
            ("KZ", "Шотыма ақша аудардым, әлі көрінбей тұр.", "Клиенттің жеке басы расталды.",
             "Аударым туралы сұрағыңызды тексеруге жолдаймын."),
            ("EN", "My transfer is missing.", "Customer identity verified.",
             "I have submitted your case for review."),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for language, issue, verified, handoff in cases:
                for scope, claimed_protected_id in (("public", None), ("account", "m2")):
                    with self.subTest(language=language, scope=scope):
                        conversation = {"language": language, "channel": "agent chat", "transcript": [
                            {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": issue},
                            {"id": "m2", "timestamp": "00:02", "speaker": "system", "text": verified},
                            {"id": "m3", "timestamp": "00:03", "speaker": "operator", "text": handoff},
                        ]}
                        self.mock_completion(opener, {"03": {
                            "scope": scope, "verification": "none", "verification_id": None,
                            "protected_id": claimed_protected_id, "evidence": {"start_id": "m1", "end_id": "m2"},
                        }})
                        [result] = evaluate(conversation, checklist)
                        self.assertIs(result["answer"], True)
                        self.assertEqual(result["evidence"], {"start_id": "m2", "end_id": "m3"})

    @patch("app.evaluation.build_opener")
    def test_late_verification_does_not_validate_individual_handoff_or_change_public_scope(self, opener):
        checklist = starter_identity_checklist()
        conversation = {"language": "EN", "channel": "agent chat", "transcript": [
            {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": "My transfer is missing."},
            {"id": "m2", "timestamp": "00:02", "speaker": "operator", "text": "I've sent the case for review."},
            {"id": "m3", "timestamp": "00:03", "speaker": "system", "text": "Customer identity verified."},
        ]}
        facts = {
            "scope": "public", "verification": "confirmed", "verification_id": "m3",
            "protected_id": "m3", "evidence": {"start_id": "m1", "end_id": "m1"},
        }
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            self.mock_completion(opener, {"03": facts})
            [result] = evaluate(conversation, checklist)
            self.assertIs(result["answer"], False)
            self.assertEqual(result["evidence"], {"start_id": "m1", "end_id": "m2"})

            conversation["transcript"][0]["text"] = "Where do I check my transfer status in the app?"
            conversation["transcript"][1]["text"] = "Customer identity verified."
            conversation["transcript"][1]["speaker"] = "system"
            conversation["transcript"][2]["text"] = "I've sent the case for review."
            conversation["transcript"][2]["speaker"] = "operator"
            self.mock_completion(opener, {"03": {**facts, "verification_id": "m2", "protected_id": "m3"}})
            [result] = evaluate(conversation, checklist)
            self.assertEqual(result["answer"], "N/A")

    @patch("app.evaluation.build_opener")
    def test_public_identity_cites_substantive_exchange_instead_of_closing(self, opener):
        checklist = starter_identity_checklist()
        cases = [
            ("EN", "Settings where? There's no login on the company site.",
             "Go to the trading site and sign in there.", "Thank you for your help.",
             "We appreciate you using our service. Good bye for now."),
            ("KZ", "Профильде ондай бөлім жоқ қой.",
             "Мәзір → Поручения → Персональные данные.", "Иә, рақмет", "Жақсы."),
            ("RU", "Какую кнопку нажать для реквизитов?", "Всё есть в приложении.",
             "Спасибо за помощь.", "До свидания."),
        ]
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            for language, question, guidance, thanks, closing in cases:
                with self.subTest(language=language):
                    conversation = {"language": language, "channel": "agent chat", "transcript": [
                        {"id": "m1", "timestamp": "00:00", "speaker": "client", "text": "Where are settings?"},
                        {"id": "m2", "timestamp": "00:01", "speaker": "operator", "text": "Open the app."},
                        {"id": "m3", "timestamp": "00:02", "speaker": "client", "text": question},
                        {"id": "m4", "timestamp": "00:03", "speaker": "operator", "text": guidance},
                        {"id": "m5", "timestamp": "00:04", "speaker": "client", "text": thanks},
                        {"id": "m6", "timestamp": "00:05", "speaker": "operator", "text": closing},
                    ]}
                    facts = {
                        "scope": "public", "verification": "none", "verification_id": None,
                        "protected_id": None, "evidence": {"start_id": "m6", "end_id": "m6"},
                    }
                    self.mock_completion(opener, {"03": facts})
                    [result] = evaluate(conversation, checklist)
                    self.assertEqual(result["answer"], "N/A")
                    self.assertEqual(result["evidence"], {"start_id": "m3", "end_id": "m4"})

                    original_exchange = {"start_id": "m1", "end_id": "m2"}
                    self.mock_completion(opener, {"03": {**facts, "evidence": original_exchange}})
                    [result] = evaluate(conversation, checklist)
                    self.assertEqual(result["evidence"], original_exchange)

    @patch("app.evaluation.build_opener")
    def test_replaced_id_03_rule_is_evaluated_generically(self, opener):
        new_rule = criterion("03", 20, na_condition="Условие для пользовательского правила.")
        self.mock_completion(opener, {"03": {
            "answer": True, "reason": "Выполнено пользовательское правило.", "evidence": {"start_id": "m2", "end_id": "m2"},
        }})
        with patch.dict(os.environ, {"CQA_LLM_RESPONSE_FORMAT": "json_schema"}):
            [result] = evaluate(self.conversation, {"criteria": [new_rule]})
        self.assertTrue(result["answer"])
        self.assertEqual(result["reason"], "Выполнено пользовательское правило.")
        body = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertNotIn("scope=public", body["messages"][0]["content"])

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
