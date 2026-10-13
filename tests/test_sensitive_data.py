import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


class SensitiveDataTests(unittest.TestCase):
    def setUp(self):
        self.original_profiles = data_service._PROFILES.copy()
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self.original_profiles)

    def test_get_prospect_returns_only_allowlisted_fields(self):
        result = gtm_agent.get_prospect.func("LEAD-50001")

        self.assertEqual(
            set(result["prospect"]),
            {"prospect_id", "name", "email", "disqualified"},
        )
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_new_profile_excludes_sensitive_fields(self):
        result = gtm_agent.build_prospect_profile.func("LEAD-50001")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES["LEAD-50001"])

    def test_profile_retrieval_sanitizes_and_replaces_cached_profile(self):
        data_service._PROFILES["lead"] = {
            "prospect_id": "lead",
            "billing_qualification": {"private": "sentinel"},
        }

        result = data_service.get_profile_from_db("lead")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES["lead"])

    def test_cached_profile_return_boundary_sanitizes(self):
        stale_profile = {
            "prospect_id": "lead",
            "billing_qualification": {"private": "sentinel"},
        }
        with patch.object(
            data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": stale_profile},
        ):
            result = gtm_agent.build_prospect_profile.func("lead")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES["lead"])

    def test_scoring_prompt_excludes_sensitive_fields_from_input(self):
        profile = {
            "prospect_id": "lead",
            "billing_qualification": {"private": "sentinel"},
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1,
            "description": "Offering",
        }
        scoring_result = unittest.mock.Mock()
        scoring_result.model_dump.return_value = {"score": 1}
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm, patch.object(
            data_service, "fetch_tech_stack", return_value=[]
        ):
            scoring_llm.invoke.return_value = scoring_result
            gtm_agent.score_prospect.func(profile, offering)

        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        self.assertNotIn("sentinel", prompt)
        self.assertIn("billing_qualification", profile)

    def test_profile_and_scoring_tool_path_stays_sanitized(self):
        prospect_result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-50001"})
        profile_result = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": "LEAD-50001"}
        )
        score_input = {
            "prospect_profile": profile_result["prospect_profile"],
            "offering": {
                "required_tech_stack": ["Snowflake"],
                "min_annual_revenue": 1,
                "description": "Offering",
            },
        }
        scoring_result = unittest.mock.Mock()
        scoring_result.model_dump.return_value = {"score": 1}
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = scoring_result
            gtm_agent.score_prospect.invoke(score_input)

        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", prospect_result)
        self.assertNotIn("billing_qualification", profile_result)
        self.assertNotIn("billing_qualification", score_input)
        self.assertNotIn("billing_qualification", prompt)


if __name__ == "__main__":
    unittest.main()
