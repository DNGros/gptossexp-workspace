from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from huggingface_hub import InferenceClient
from enum import StrEnum
import diskcache
from typing import Optional

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
    
    return ModelResponse(
        response=content,
        reasoning="",  # Pipeline doesn't expose reasoning separately
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def get_model_response(
    model: Model,
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    use_cache: bool = True,
    backend: InferenceBackend = InferenceBackend.API,
):
    """
    Get model response using specified backend.
    
    Args:
        model: Model to use
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        use_cache: Whether to use disk cache
        backend: Inference backend (API or LOCAL)
    
    Returns:
        ModelResponse with response text, reasoning, and metadata
    """
    # Check cache first
    cache_key = _get_cache_key(model, prompt, system_prompt, temperature, max_tokens, backend)
    cached_response = cache.get(cache_key)
    if cached_response is not None and use_cache:
        return cached_response
    
    # Route to appropriate backend
    if backend == InferenceBackend.LOCAL:
        response = _get_local_model_response(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:  # API backend
        # Make API call
        client = InferenceClient()
        messages = _build_messages(prompt, system_prompt)
        
        # Build parameters dict, only including non-None values
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