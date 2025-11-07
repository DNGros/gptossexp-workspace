from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from huggingface_hub import InferenceClient
from enum import StrEnum
import diskcache
from typing import Optional, Union, List

class Model(StrEnum):
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_safeguarded_20B = "openai/gpt-oss-safeguard-20b"


class InferenceBackend(StrEnum):
    """Backend for model inference."""
    API = "api"          # Hugging Face Inference API (default)
    LOCAL = "local"      # Local transformers with apply_chat_template


# Cache directory for storing responses
CACHE_DIR = Path(".cache/model_responses")
cache = diskcache.Cache(str(CACHE_DIR))


class LocalModelCache:
    """Singleton cache for local pipeline generators to avoid reloading."""
    _instance = None
    _pipelines = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pipeline(self, model_id: str):
        """Get or load text-generation pipeline."""
        if model_id not in self._pipelines:
            print(f"Loading model pipeline: {model_id} (this may take a while...)")
            
            # Lazy import to avoid loading transformers unless needed
            from transformers import pipeline
            
            self._pipelines[model_id] = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype="auto",
                device_map="auto",  # Automatically place on available GPUs
            )
            print(f"Pipeline loaded successfully")
        
        return self._pipelines[model_id]
    
    def clear(self):
        """Clear cached pipelines to free memory."""
        self._pipelines.clear()


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
    backend: InferenceBackend = InferenceBackend.API,
) -> str:
    """Generate a cache key from the input parameters."""
    cache_data = {
        "model": str(model),
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if backend != InferenceBackend.API:
        # For backwards compatibility only add if new endpoint
        cache_data["backend"] = str(backend)
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
    """
    Build messages list for chat completion.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
    
    Returns:
        List of message dicts in chat format
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _get_local_model_response(
    model: Model,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> ModelResponse:
    """
    Get model response using local transformers pipeline.
    
    Uses the high-level pipeline API which handles chat templates automatically.
    """
    # Get cached pipeline
    model_cache = LocalModelCache()
    generator = model_cache.get_pipeline(str(model))
    
    # Build messages
    messages = _build_messages(prompt, system_prompt)
    
    # Generate response using pipeline
    result = generator(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1.0,  # pipeline requires temp > 0
        do_sample=(temperature > 0),
    )
    
    # Extract the generated text
    # The pipeline returns the full conversation including the prompt,
    # so we need to get just the assistant's response
    generated_text = result[0]["generated_text"]
    
    # The last message in generated_text should be the assistant's response
    if isinstance(generated_text, list) and len(generated_text) > len(messages):
        assistant_message = generated_text[-1]
        content = assistant_message.get("content", "")
    else:
        # Fallback: use the whole generated text
        content = str(generated_text)

    # Hacky parse the response.
    reasoning, content = content.rsplit("final", 1)
    if reasoning.startswith("analysis"):
        reasoning = reasoning[len("analysis"):]
    
    return ModelResponse(
        response=content,
        reasoning=reasoning,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def _get_local_model_response_batch(
    model: Model,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> List[ModelResponse]:
    """
    Get model responses for a batch of prompts using local transformers pipeline.

    Uses the high-level pipeline API which handles chat templates and batching automatically.
    """
    # Get cached pipeline
    model_cache = LocalModelCache()
    generator = model_cache.get_pipeline(str(model))

    # Build messages for all prompts
    messages_batch = [_build_messages(prompt, system_prompt) for prompt in prompts]

    # Generate responses using pipeline with batching
    results = generator(
        messages_batch,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        do_sample=(temperature > 0),
        batch_size=len(prompts),  # Explicit batch size
    )

    # Parse results into ModelResponse objects
    model_responses = []
    for prompt, result in zip(prompts, results):
        # Extract the generated text
        generated_text = result[0]["generated_text"]

        # The last message in generated_text should be the assistant's response
        if isinstance(generated_text, list) and len(generated_text) > len(_build_messages(prompt, system_prompt)):
            assistant_message = generated_text[-1]
            content = assistant_message.get("content", "")
        else:
            # Fallback: use the whole generated text
            content = str(generated_text)

        # Hacky parse the response
        reasoning, content = content.rsplit("final", 1)
        if reasoning.startswith("analysis"):
            reasoning = reasoning[len("analysis"):]

        model_responses.append(ModelResponse(
            response=content,
            reasoning=reasoning,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
        ))

    return model_responses


def get_model_response(
    model: Model,
    prompt: Union[str, List[str]],
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    use_cache: bool = True,
    backend: InferenceBackend = InferenceBackend.API,
) -> Union[ModelResponse, List[ModelResponse]]:
    """
    Get model response using specified backend.

    Accepts either a single prompt or a list of prompts.
    For LOCAL backend with batched prompts, uses efficient batching with cache filtering.

    Args:
        model: Model to use
        prompt: User prompt (single string or list of strings)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        use_cache: Whether to use disk cache
        backend: Inference backend (API or LOCAL)

    Returns:
        ModelResponse (single) or List[ModelResponse] (batch)
    """
    # Detect if batch or single
    is_batch = isinstance(prompt, list)
    prompts = prompt if is_batch else [prompt]
    n = len(prompts)

    # Step 1: Check cache for all prompts
    cache_keys = [
        _get_cache_key(model, p, system_prompt, temperature, max_tokens, backend)
        for p in prompts
    ]
    cached_results = [cache.get(key) if use_cache else None for key in cache_keys]

    # Step 2: Identify cache misses
    miss_indices = [i for i, cached in enumerate(cached_results) if cached is None]
    miss_prompts = [prompts[i] for i in miss_indices]

    if is_batch and miss_prompts:
        print(f"Cache: {n - len(miss_indices)} hits, {len(miss_indices)} misses")

    # Step 3: Get responses for cache misses
    miss_responses = []
    if miss_prompts:
        if backend == InferenceBackend.LOCAL and len(miss_prompts) > 1:
            # Batch inference for LOCAL backend
            miss_responses = _get_local_model_response_batch(
                model=model,
                prompts=miss_prompts,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Sequential processing for API backend or single miss
            for miss_prompt in miss_prompts:
                if backend == InferenceBackend.LOCAL:
                    response = _get_local_model_response(
                        model=model,
                        prompt=miss_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:  # API backend
                    client = InferenceClient()
                    messages = _build_messages(miss_prompt, system_prompt)

                    params = {}
                    if temperature is not None:
                        params["temperature"] = temperature
                    if max_tokens is not None:
                        params["max_tokens"] = max_tokens

                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **params,
                    )
                    response = ModelResponse(
                        response=completion.choices[0].message.content,
                        reasoning=getattr(completion.choices[0].message, 'reasoning', None) or "",
                        model=model,
                        prompt=miss_prompt,
                        system_prompt=system_prompt,
                    )

                miss_responses.append(response)

        # Cache the new results
        for idx, response in zip(miss_indices, miss_responses):
            cache.set(cache_keys[idx], response)

    # Step 4: Merge cached + new in original order
    results = []
    miss_iter = iter(miss_responses)
    for i in range(n):
        if cached_results[i] is not None:
            results.append(cached_results[i])
        else:
            results.append(next(miss_iter))

    # Return single or batch based on input
    return results if is_batch else results[0]


if __name__ == "__main__":
    response = get_model_response(
        model=Model.GPT_OSS_20B,
        prompt="How many 'G's in 'huggingface'?",
        use_cache=False,
    )
    print(response)