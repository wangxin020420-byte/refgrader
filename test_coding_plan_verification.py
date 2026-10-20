import tempfile
import unittest
from pathlib import Path

from scripts.verify_coding_plan import EXPECTED_BASE_URL, SOURCE_FILES, audit_sources, validate_base_url


class CodingPlanVerificationTests(unittest.TestCase):
    def test_only_exact_coding_endpoint_is_allowed(self):
        validate_base_url(EXPECTED_BASE_URL)
        validate_base_url(EXPECTED_BASE_URL + "/")
        for url in (
            "https://open.bigmodel.cn/api/paas/v4", "http://open.bigmodel.cn/api/coding/paas/v4",
            EXPECTED_BASE_URL + "?route=normal", EXPECTED_BASE_URL + "//",
            "https://open.bigmodel.cn.evil.invalid/api/coding/paas/v4", None,
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_base_url(url)

    def test_default_endpoint_client_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / SOURCE_FILES[0]).write_text("client = OpenAI(api_key='fake')", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "default endpoint"):
                audit_sources(root)

    def test_new_client_requires_coverage_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / SOURCE_FILES[0]).write_text("a = OpenAI()\nb = OpenAI()", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage changed"):
                audit_sources(root)

    def test_current_api_entry_points_have_explicit_endpoints(self):
        report = audit_sources()
        self.assertEqual(sum(row["explicit_client_constructors"] for row in report), 7)


if __name__ == "__main__":
    unittest.main()
