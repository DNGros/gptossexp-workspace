from dataclasses import dataclass
import re
from pathlib import Path
from .model_predict import ModelResponse, get_model_response, Model, InferenceBackend


@dataclass
class ClassificationResult:
    binary_label: bool  # True = violates policy, False = ok
    fine_grain_label: dict | str | None
    metadata: dict
    model_response: ModelResponse
    parsed_successfully: bool  # True if parser successfully extracted meaningful information


def _default_parse(response: ModelResponse) -> ClassificationResult:
    """Default parser when policy module doesn't provide one."""
    text = response.response
    
    # Extract binary label - INVALID means violates (True)
    binary_label = "INVALID" in text.upper()
    
    # Try to extract fine-grain labels
    fine_grain = None
    
    # Look for patterns like D-SP2, R-SP0, SP3, etc.
    sp_matches = re.findall(r'[DR]-?SP\d+(?:\.[a-z])?', text, re.IGNORECASE)
    if sp_matches:
        # Check if we have both D- and R- labels
        d_labels = [m for m in sp_matches if m.upper().startswith('D')]
        r_labels = [m for m in sp_matches if m.upper().startswith('R')]
        if d_labels or r_labels:
            fine_grain = {
                "depiction": d_labels[0] if d_labels else None,
                "request": r_labels[0] if r_labels else None
            }
        else:
            fine_grain = sp_matches[0] if sp_matches else None
    
    # Parsing was successful if we found INVALID keyword or SP labels
    parsed_successfully = binary_label or (fine_grain is not None)
    
    metadata = {}
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=fine_grain,
        metadata=metadata,
        model_response=response,
        parsed_successfully=parsed_successfully
    )


def classify(
    text: str,
    policy_module,
    model: Model = Model.GPT_OSS_20B,
    backend: InferenceBackend = InferenceBackend.API,
    use_cache: bool = True,
) -> ClassificationResult:
    """
    Classify text using a policy module.
    
    Args:
        text: Text to classify
        policy_module: Python module from policies/ (e.g., policies.spam)
        model: Model to use for classification
        backend: Inference backend (API or LOCAL)
    
    Returns:
        ClassificationResult with binary label, fine-grain label, and metadata
    """
    # Load the markdown file from the policy module's directory
    module_path = Path(policy_module.__file__).parent
    module_name = policy_module.__name__.split('.')[-1]
    md_path = module_path / f"{module_name}.md"
    
    if not md_path.exists():
        raise FileNotFoundError(f"Policy file not found: {md_path}")
    
    system_prompt = md_path.read_text()
    
    # Get model response
    response = get_model_response(
        model=model,
        prompt=text,
        system_prompt=system_prompt,
        backend=backend,
        use_cache=use_cache,
    )
    
    # Use custom parser if available, otherwise default
    if hasattr(policy_module, 'parse'):
        return policy_module.parse(response)
    else:
        return _default_parse(response)

