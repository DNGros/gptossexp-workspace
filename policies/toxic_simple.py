import re
from ..model_predict import ModelResponse
from ..classify import ClassificationResult
from . import toxic_chat_claude_1


def parse(response: ModelResponse) -> ClassificationResult:
    return toxic_chat_claude_1.parse(response)
