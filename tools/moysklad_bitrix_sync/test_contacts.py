import unittest

from contacts import best_match, extract_phones, match_score, normalize_phone


class NormalizePhoneTest(unittest.TestCase):
    def test_russian_formats(self):
        for raw in ("89286435976", "8928 643 59 76", "+7 928 643-59-76", "9286435976", "8 (928) 643-59-76"):
            self.assertEqual(normalize_phone(raw), "+79286435976", raw)

    def test_foreign_number_keeps_country_code(self):
        self.assertEqual(normalize_phone("+33 621 27 27 28"), "+33621272728")

    def test_not_a_phone(self):
        for raw in ("", None, "188380", "8-9", "12345678901"):
            self.assertIsNone(normalize_phone(raw), raw)


class ExtractPhonesTest(unittest.TestCase):
    def test_phone_in_counterparty_name(self):
        self.assertEqual(extract_phones("Магомадов Адлан  8-920-092-46-42"), ["+79200924642"])
        self.assertEqual(extract_phones("Шаипова Луиза 7 938 022 22 52"), ["+79380222252"])
        self.assertEqual(extract_phones("Якъуб Гудермес    +7 928 002-07-67"), ["+79280020767"])

    def test_two_phones_glued_together(self):
        self.assertEqual(
            extract_phones("89204222289  89288888389"), ["+79204222289", "+79288888389"]
        )
        self.assertEqual(
            extract_phones("Ваха Деши 89288910411,89286444964"), ["+79288910411", "+79286444964"]
        )

    def test_amounts_and_sizes_are_not_phones(self):
        comment = "ВП 7\nоплатила перечислением 985.179\n297.513 на викторию счет 538\n687.666 на фадекс счет 13"
        self.assertEqual(extract_phones(comment), [])
        self.assertEqual(extract_phones("Дверь Жовнер 1900*2800, доплата 34.321"), [])
        self.assertEqual(extract_phones("Хеда Шерипова  8-9"), [])


class MatchTest(unittest.TestCase):
    order = {
        "name": "Гадамаури Анна Магомедовна",
        "phones": ["8928 643 59 76"],
        "address": "364052, Россия, Чеченская Респ, г Грозный, ул Ключевая, 142",
        "sum": 1273508,
    }

    def test_shared_phone_is_decisive(self):
        deal = {"name": "Анна Ключевая", "phones": ["+79286435976"], "sum": 5}
        self.assertEqual(match_score(self.order, deal)["score"], 100)

    def test_name_address_and_sum_match_without_phone(self):
        deal = {"name": "Гадамаури Анна", "address": "Грозный, Ключевая 142", "sum": 1270000}
        result = best_match(self.order, [deal, {"name": "Петимат", "sum": 100}])
        self.assertEqual(result["decision"], "auto")

    def test_name_only_needs_manual_review(self):
        deal = {"name": "Гадамаури Анна", "sum": 500000}
        self.assertEqual(best_match(self.order, [deal])["decision"], "no_match")
        deal["address"] = "ул Ключевая 142"
        self.assertEqual(best_match(self.order, [deal])["decision"], "review")

    def test_two_equal_candidates_are_not_auto(self):
        deal = {"name": "Гадамаури Анна", "address": "Ключевая 142", "sum": 1273508}
        self.assertEqual(best_match(self.order, [deal, dict(deal)])["decision"], "review")


if __name__ == "__main__":
    unittest.main()
