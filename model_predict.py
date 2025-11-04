from dataclasses import dataclass
import os
from huggingface_hub import InferenceClient
from enum import StrEnum

class Model(StrEnum):
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_safeguarded_20B = "openai/gpt-oss-safeguard-20b"


@dataclass
class ModelResponse:
    response: str
    reasoning: str
    model: Model
    prompt: str
    system_prompt: str


def get_model_response(
    model: Model,
    prompt: str,
    system_prompt: str = None,
):
    client = InferenceClient()
    msgs = []
    if system_prompt:
        msgs.append({
            "role": "system",
            "content": system_prompt
        })
    msgs.append({
        "role": "user",
        "content": prompt
    })
    completion = client.chat.completions.create(
        model=model,
        messages=msgs,
    )
    return ModelResponse(
        response=completion.choices[0].message.content,
        reasoning=completion.choices[0].message.reasoning,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    response = get_model_response(
        model=Model.GPT_OSS_20B,
        prompt="How many 'G's in 'huggingface'?",
    )
    print(response)