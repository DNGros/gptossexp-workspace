import os
from huggingface_hub import InferenceClient


def main():
    client = InferenceClient()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "How many 'G's in 'huggingface'?"
            }
        ],
    )
    print(completion.choices[0].message)


if __name__ == "__main__":
    main()