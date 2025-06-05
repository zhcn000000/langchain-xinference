"""Test chat model integration."""

from typing import Type

from langchain_xinference.chat_models import ChatXinference
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatXinferenceUnit(ChatModelUnitTests):
    """
    test
    """
    @property
    def chat_model_class(self) -> Type[ChatXinference]:
        return ChatXinference

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
