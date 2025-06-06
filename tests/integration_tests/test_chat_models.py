"""Test ChatXinference chat model."""

from typing import Type

from langchain_xinference.chat_models import ChatXinference
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatXinference]:
        return ChatXinference

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
