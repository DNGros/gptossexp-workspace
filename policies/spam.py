import re
from ..model_predict import ModelResponse
from ..classify import ClassificationResult


def parse(response: ModelResponse) -> ClassificationResult:
    """Parse spam policy response with D-SP# and R-SP# labels."""
    text = response.response
    
    # Extract D-SP# (depiction) and R-SP# (request) labels
    d_match = re.search(r'D-SP(\d+)(?:\.([a-z]))?', text, re.IGNORECASE)
    r_match = re.search(r'R-SP(\d+)(?:\.([a-z]))?', text, re.IGNORECASE)
    
    fine_grain = {
        "depiction": d_match.group(0).upper() if d_match else None,
        "request": r_match.group(0).upper() if r_match else None,
    }
    
    # Binary label: INVALID if explicitly stated, OR if D-SP level is 2+ (spam detected)
    binary_label = "INVALID" in text.upper()
    if not binary_label and d_match:
        # D-SP0 = not spam, D-SP2+ = spam
        d_level = int(d_match.group(1))
        binary_label = d_level >= 2
    
    metadata = {}
    
    return ClassificationResult(
        binary_label=binary_label,
        fine_grain_label=fine_grain,
        metadata=metadata,
        model_response=response
    )

