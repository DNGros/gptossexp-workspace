from workspace.classify import classify
from workspace.model_predict import InferenceBackend, Model
from workspace.policies import toxic_simple


def hello_controversial():
    opinion = "Giving your kid an iPad is lazy parenting, full stop."
    
    print(f"Opinion to classify: {opinion}\n")
    
    # Classify the opinion using the toxic_simple policy
    result = classify(
        text=opinion,
        policy_module=toxic_simple,
        model=Model.GPT_OSS_safeguarded_20B,
        backend=InferenceBackend.API,
    )
    
    print("Classification Results:")
    print(f"  Binary Label: {'TOXIC' if result.binary_label else 'NON-TOXIC'}")
    print(f"  Fine-grain Label: {result.fine_grain_label}")
    print(f"  Parsed Successfully: {result.parsed_successfully}")
    print(f"\nModel Response:")
    print(f"  {result.model_response.response}")
    
    return result


if __name__ == "__main__":
    hello_controversial()