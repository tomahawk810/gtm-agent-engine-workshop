import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-001"}})

    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Follow up",
            body="Hello",
            runtime=self.runtime,
            from_rep={"name": "Alex Rep", "email": "alex@example.com"},
        )

    def test_disqualified_id_blocks_before_message_id(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
        }
        with patch("gtm_agent.gtm_agent.uuid.uuid4") as make_uuid:
            result = self.send(prospect)

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is marked disqualified; email not sent."},
        )
        make_uuid.assert_not_called()

    def test_prompt_requires_explaining_block_without_claiming_send(self):
        self.assertIn("marked disqualified or the send is blocked", SYSTEM_PROMPT)
        self.assertIn("tell the rep why", SYSTEM_PROMPT)
        self.assertIn("never claim it was sent", SYSTEM_PROMPT)
        self.assertNotIn("do not second-guess", SYSTEM_PROMPT)

    def test_non_disqualified_prospect_is_sent(self):
        result = self.send({
            "prospect_id": "LEAD-12853",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
        })

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_email_lookup_blocks_disqualified_prospect_without_id(self):
        result = self.send({
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
        })

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["error"], "Prospect is marked disqualified; email not sent.")

    def test_missing_email_preserves_existing_failure(self):
        self.assertEqual(
            self.send({}),
            {"status": "failed", "error": "Prospect record has no email address."},
        )


if __name__ == "__main__":
    unittest.main()
