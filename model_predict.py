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
    """Singleton cache for local models and tokenizers to avoid reloading."""
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model_and_tokenizer(self, model_id: str):
        """Get or load model and tokenizer."""
        if model_id not in self._models:
            print(f"Loading model: {model_id} (this may take a while...)")
            
            # Lazy import to avoid loading transformers/torch unless needed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
            self._models[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
            )
            print(f"Model loaded on device: {next(self._models[model_id].parameters()).device}")
        
        return self._models[model_id], self._tokenizers[model_id]
    
    def clear(self):
        """Clear cached models to free memory."""
        self._models.clear()
        self._tokenizers.clear()


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


def _parse_harmony_channels(raw_output: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse Harmony format channels from model output.
    
    Returns:
        tuple: (reasoning, content) where reasoning is from analysis channel
               and content is from final channel
    """
    reasoning = None
    content = None
    
    # Look for analysis channel
    if '<|channel|>analysis' in raw_output:
        parts = raw_output.split('<|channel|>')
        for part in parts:
            if part.startswith('analysis'):
                if '<|message|>' in part:
                    analysis_content = part.split('<|message|>', 1)[1]
                    # Stop at next special token
                    analysis_content = analysis_content.split('<|')[0]
                    reasoning = analysis_content.strip()
    
    # Look for final channel
    if '<|channel|>final' in raw_output:
        parts = raw_output.split('<|channel|>')
        for part in parts:
            if part.startswith('final'):
                if '<|message|>' in part:
                    final_content = part.split('<|message|>', 1)[1]
                    final_content = final_content.split('<|')[0]
                    content = final_content.strip()
    
    return reasoning, content


def _get_local_model_response(
    model: Model,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> ModelResponse:
    """
    Get model response using local transformers with apply_chat_template.
    
    This applies the Harmony format via the model's chat template.
    """
    # Lazy import to avoid loading unless needed
    import torch
    
    # Get cached model and tokenizer
    model_cache = LocalModelCache()
    local_model, tokenizer = model_cache.get_model_and_tokenizer(str(model))
    
    # Build messages
    messages = _build_messages(prompt, system_prompt)
    
    # Apply chat template (this should apply Harmony format)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(local_model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = local_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract just the new tokens (the response)
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw_output = tokenizer.decode(response_tokens, skip_special_tokens=False)
    
    # Try to parse Harmony format channels
    reasoning, content = _parse_harmony_channels(raw_output)
    
    # Fallback: if we can't parse channels, use the whole response with special tokens removed
    if content is None:
        content = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print("Warning: Could not parse Harmony channels, using full decoded response")
    
    return ModelResponse(
        response=content,
        reasoning=reasoning or "",
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