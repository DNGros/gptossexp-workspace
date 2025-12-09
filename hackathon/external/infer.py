"""
Copied from https://huggingface.co/spaces/roosttools/llm_moderation_testing/raw/main/utils/model_interface.py

Model interface for calling moderation models."""

import json
import re

from openai import OpenAI
from openai_harmony import (
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)

from utils.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GENERIC_SYSTEM_PROMPT_PREFIX,
    LM_PROMPT_INSTRUCT,
    COPE_PROMPT_TEMPLATE,
    RESPONSE_FORMAT,
    ROUTER_URL,
    MODELS,
)


def get_model_info(model_id: str) -> dict:
    """Get model metadata by ID."""
    for model in MODELS:
        if model["id"] == model_id:
            return model
    return None


def extract_model_id(choice: str) -> str:
    """Extract model ID from dropdown choice format 'Name (id)'."""
    if not choice:
        return ""
    return choice.split("(")[-1].rstrip(")")


def is_gptoss_model(model_id: str) -> bool:
    """Check if model is GPT-OSS."""
    return model_id.startswith("openai/gpt-oss")


def get_default_system_prompt(model_id: str, reasoning_effort: str = "Low") -> str:
    """Generate default system prompt based on model type and policy."""
    if is_gptoss_model(model_id):
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        system_prompt_harmony = Message.from_role_and_content(
            Role.SYSTEM, SystemContent.new().with_reasoning_effort(reasoning_effort)
        )
        system_prompt_dict = enc.decode(enc.render(system_prompt_harmony))
        system_prompt_content = re.search(r"<\|message\|>(.*?)<\|end\|>", system_prompt_dict, re.DOTALL).group(1)
        return system_prompt_content
    else:
        # Qwen: formatted system prompt (goes in system role)
        return GENERIC_SYSTEM_PROMPT_PREFIX


def make_messages(
    test: str, 
    policy: str, 
    model_id: str, 
    reasoning_effort: str = "Low", 
    system_prompt: str | None = None, 
    response_format: str = RESPONSE_FORMAT
) -> list[dict]:
    """Create messages based on model type."""
    if model_id.startswith("openai/gpt-oss-safeguard"):
        # GPT-OSS uses Harmony encoding
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        system_content = SystemContent.new().with_reasoning_effort(reasoning_effort)
        conv_messages = [
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(policy + "\n\n" + response_format),
            ),
            Message.from_role_and_content(Role.USER, test),
        ]
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for pre_msg in conv_messages:
            tokens = enc.render(pre_msg)
            prompt = enc.decode(tokens)
            messages.append({
                "role": re.search(r"<\|start\|>(.*?)<\|message\|>", prompt).group(1),
                "content": re.search(r"<\|message\|>(.*?)<\|end\|>", prompt, re.DOTALL).group(1),
            })
        return messages
    else:
        system_content = LM_PROMPT_INSTRUCT.format(
            system_prompt=system_prompt,
            policy=policy,
            response_format=response_format
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Content: {test}\n\nResponse:"},
        ]


def run_test(
    model_id: str,
    test_input: str,
    policy: str,
    hf_token: str,
    reasoning_effort: str = "Low",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    system_prompt: str | None = None,
    response_format: str = RESPONSE_FORMAT,
) -> dict:
    """Run test on model."""
    import gradio as gr
    
    model_info = get_model_info(model_id)
    if not model_info:
        raise gr.Error(f"Unknown model: {model_id}. Please select a valid model from the dropdown.")

    client = OpenAI(base_url=ROUTER_URL, api_key=hf_token)

    # CoPE special case as a completions-only model
    if model_id.startswith("zentropi-ai/cope-a-9b"):
        try:
            from utils.helpers import get_cope_token
            cope_token = get_cope_token()
            client = OpenAI(base_url='https://q0c7gn9b2s2wkb6v.us-east-1.aws.endpoints.huggingface.cloud/v1', api_key=cope_token)
            cope_prompt = COPE_PROMPT_TEMPLATE.format(policy=policy,content=test_input)
            response = client.completions.create(
                model='cope-a-9b',
                prompt=cope_prompt,
                max_tokens=1,
                temperature=0,
            )
            result_content = f"""```json
{{
  "label": {response.choices[0].text},
  "categories": []
}}
"""
            result = {"content": result_content}
            return result
        except Exception as e:
            error_msg = str(e)
            raise gr.Error(f"Model inference failed: {error_msg}. Sorry, it was a last minute hack to add this model to this interface!")
        
    messages = make_messages(test_input, policy, model_id, reasoning_effort, system_prompt, response_format)

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=None,
            extra_headers={"X-HF-Bill-To": "roosttools"},
        )
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            raise gr.Error(f"Authentication failed: {error_msg}. Please check your token permissions.")
        elif "400" in error_msg or "bad request" in error_msg.lower():
            raise gr.Error(f"Invalid request: {error_msg}. Please check your input and try again.")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise gr.Error(f"Rate limit exceeded: {error_msg}. Please wait a moment and try again.")
        elif "timeout" in error_msg.lower():
            raise gr.Error(f"Request timed out: {error_msg}. Please try again with a shorter input or lower max_tokens.")
        else:
            raise gr.Error(f"Model inference failed: {error_msg}. Please check your inputs and try again.")

    result = {"content": completion.choices[0].message.content}

    # Extract reasoning if available
    message = completion.choices[0].message
    if model_info["is_thinking"]:
        if is_gptoss_model(model_id):
            # GPT-OSS: check reasoning or reasoning_content field
            if hasattr(message, "reasoning") and message.reasoning:
                result["reasoning"] = message.reasoning
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                result["reasoning"] = message.reasoning_content
        else:
            # Qwen Thinking: extract from content using </think> tag
            content = message.content
            if "</think>" in content:
                result["reasoning"] = content.split("</think>")[0].strip()
                # Also update content to be the part after </think>
                result["content"] = content.split("</think>")[-1].strip()

    return result


