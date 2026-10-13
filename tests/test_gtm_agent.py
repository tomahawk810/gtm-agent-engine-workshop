import importlib
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch


class GtmAgentTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test")
        self.agent = Mock()
        self.create_agent = Mock(return_value=self.agent)
        module = importlib.import_module("gtm_agent.gtm_agent")
        self.compiled_agent = module.gtm_agent
        with patch("langchain.agents.create_agent", self.create_agent):
            self.module = importlib.reload(module)

    def test_agent_uses_only_domain_tools_and_application_prompt(self):
        self.create_agent.assert_called_once()
        arguments = self.create_agent.call_args.kwargs
        self.assertIs(arguments["model"], self.module.agent_model)
        self.assertEqual(
            [tool.name for tool in arguments["tools"]],
            [
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            ],
        )
        self.assertEqual(arguments["system_prompt"], self.module.SYSTEM_PROMPT)
        self.assertNotIn("middleware", arguments)
        exposed_tools = self.compiled_agent.get_graph().nodes["tools"].data.tools_by_name
        self.assertEqual(set(exposed_tools), set(tool.name for tool in arguments["tools"]))

    def test_run_agent_preserves_invoke_configuration(self):
        run_id = uuid.UUID("ae5a41e2-dad7-4c77-b65c-52a3f105d4aa")
        thread_id = "thread-123"
        self.agent.invoke.return_value = {
            "messages": [SimpleNamespace(content="done")],
        }

        with patch.object(self.module.uuid, "uuid4", return_value=run_id):
            result = self.module.run_agent(
                "Score prospect LEAD-12345",
                user_id="rep-123",
                environment="staging",
                thread_id=thread_id,
            )

        self.agent.invoke.assert_called_once()
        arguments = self.agent.invoke.call_args
        self.assertEqual(arguments.kwargs["config"]["run_name"], "GTM Assistant")
        self.assertIs(arguments.kwargs["config"]["run_id"], run_id)
        self.assertEqual(
            arguments.kwargs["config"]["metadata"],
            {
                "thread_id": thread_id,
                "user_id": "rep-123",
                "environment": "staging",
                "request_intent": "score_prospect",
            },
        )
        self.assertEqual(result["run_id"], str(run_id))
