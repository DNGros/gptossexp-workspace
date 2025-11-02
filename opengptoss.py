"""
Download and load the gpt-oss-20b model from Hugging Face.
"""
print("importing transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("importing torch")
import torch

def download_gpt_oss_20b():
    """
    Download the gpt-oss-20b model from Hugging Face.
    The model will be cached locally after the first download.
    """
    model_id = "openai/gpt-oss-20b"
    
    print(f"Downloading model: {model_id}")
    print("This may take a while depending on your internet connection...")
    
    # Download and load the tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Download and load the model
    # Using dtype="auto" to automatically select the best dtype
    # Using device_map="cpu" to force CPU usage
    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    
    print(f"Model downloaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer

def hello_world_completion(model, tokenizer):
    """
    Generate a simple "hello world" completion using the model.
    """
    # Prepare the messages in the format expected by the harmony response format
    messages = [
        {"role": "user", "content": "Say hello world!"},
    ]
    
    # Apply the chat template (automatically handles harmony format)
    print("\nGenerating response...")
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    return response

if __name__ == "__main__":
    model, tokenizer = download_gpt_oss_20b()
    print("\nModel and tokenizer are ready to use!")
    
    # Generate a hello world completion
    hello_world_completion(model, tokenizer)

