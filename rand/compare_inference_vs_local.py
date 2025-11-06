"""
Minimal reproducible example comparing InferenceClient vs transformers pipeline.
"""

from huggingface_hub import InferenceClient
from transformers import pipeline

# Simple prompt
messages = [{"role": "user", "content": "Give advice on planning a trip to Paris."}]

print("API (InferenceClient)")
print("="*70)

# Call API
client = InferenceClient()
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    temperature=0.0,
    max_tokens=200,
    #extra_body={"reasoning_effort": "medium"}
)

# Print reasoning and output
message = completion.choices[0].message
print("REASONING:")
print(message.reasoning)
print("\nOUTPUT:")
print(message.content)

print("\n" + "="*70)
print("LOCAL (transformers pipeline)")
print("="*70)

# Call pipeline
generator = pipeline("text-generation", model="openai/gpt-oss-20b", torch_dtype="auto", device_map="auto")
result = generator(
    messages,
    max_new_tokens=200,
    temperature=0.0,
    do_sample=False,
)

# Print output
response = result[0]["generated_text"][-1]["content"]
print("OUTPUT:")
print(response)
