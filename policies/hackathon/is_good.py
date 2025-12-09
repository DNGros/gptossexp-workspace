import re
from workspace.classify import ClassificationResult
from workspace.model_predict import ModelResponse
from workspace.policies.hackathon.parse_likert import parse_likert


def parse(response: ModelResponse) -> ClassificationResult:
    return parse_likert(response, min_value=1, max_value=5)
