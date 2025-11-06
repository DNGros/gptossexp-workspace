from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from huggingface_hub import InferenceClient
from enum import StrEnum
import diskcache

class Model(StrEnum):
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_safeguarded_20B = "openai/gpt-oss-safeguard-20b"


# Cache directory for storing responses
CACHE_DIR = Path(".cache/model_responses")
cache = diskcache.Cache(str(CACHE_DIR))


@dataclass
class ModelResponse:
    response: str
    reasoning: str
    model: Model
    prompt: str
    system_prompt: str


def _get_cache_key(
    model: Model, 
    prompt: str, 
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """Generate a cache key from the input parameters."""
    cache_data = {
        "model": str(model),
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()


def get_model_response(
    model: Model,
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    use_cache: bool = True,
):
    # Check cache first
    cache_key = _get_cache_key(model, prompt, system_prompt, temperature, max_tokens)
    cached_response = cache.get(cache_key)
    if cached_response is not None and use_cache:
        return cached_response
    
    # If not cached, make API call
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
    
    # Build parameters dict, only including non-None values
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    
    completion = client.chat.completions.create(
        model=model,
        messages=msgs,
        **params,
    )
    response = ModelResponse(
        response=completion.choices[0].message.content,
        reasoning=getattr(completion.choices[0].message, 'reasoning', None) or "",
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    
    # Save to cache
    cache.set(cache_key, response)
    return response


if __name__ == "__main__":
    response = get_model_response(
        model=Model.GPT_OSS_20B,
        prompt="How many 'G's in 'huggingface'?",
        use_cache=False,
    )
    print(response)