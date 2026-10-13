import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_agent import (
    ProspectScore,
    RubricBreakdown,
    build_prospect_profile,
    score_prospect,
)


class ProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.record = data_service.PROSPECTS[self.prospect_id]
        self.original_tech_stack = list(self.record["tech_stack"])
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        self.record["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.pop(self.prospect_id, None)

    def test_update_persists_technology_and_invalidates_cached_profile(self):
        initial = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", initial["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", result["tech_stack"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        refreshed = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", refreshed["prospect_profile"]["tech_stack"])

    def test_rescoring_uses_updated_saved_technology(self):
        profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        offering = {
            "required_tech_stack": ["Terraform"],
            "min_annual_revenue": 1,
            "description": "Infrastructure management",
        }

        def score_updated_profile(messages):
            prospect = json.loads(
                messages[1]["content"].split("\n\nProspect profile:\n", 1)[1]
            )
            has_technology = "Terraform" in prospect["tech_stack"]
            return ProspectScore(
                score=100 if has_technology else 0,
                justification=(
                    "Terraform is present."
                    if has_technology
                    else "Terraform is missing."
                ),
                rubric_breakdown=RubricBreakdown(
                    revenue_fit=100,
                    tech_stack_match=100 if has_technology else 0,
                    segment_fit=100,
                ),
            )

        with patch.object(
            agent_module,
            "_scoring_llm",
            SimpleNamespace(invoke=score_updated_profile),
        ):
            result = score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        self.assertNotIn("missing", result["justification"].lower())
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)


if __name__ == "__main__":
    unittest.main()
