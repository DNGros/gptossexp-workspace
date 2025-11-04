import pytest
from workspace.classify import classify, _default_parse, ClassificationResult
from workspace.model_predict import ModelResponse, Model
from workspace.policies import spam


def test_default_parse_invalid():
    """Test default parser extracts INVALID and labels."""
    response = ModelResponse(
        response="This is spam. Classification: D-SP2, R-SP0. Output: INVALID",
        reasoning="Contains promotional content",
        model=Model.GPT_OSS_20B,
        prompt="test",
        system_prompt="test"
    )
    
    result = _default_parse(response)
    
    assert result.binary_label == True  # INVALID = violates
    assert result.fine_grain_label == {"depiction": "D-SP2", "request": "R-SP0"}
    assert result.model_response == response


def test_spam_policy_parse():
    """Test spam policy custom parser."""
    response = ModelResponse(
        response="Analysis: This is cold promotion. D-SP2.a, R-SP0. INVALID",
        reasoning="Unsolicited promotional content",
        model=Model.GPT_OSS_20B,
        prompt="Check out my course!",
        system_prompt="spam policy"
    )
    
    result = spam.parse(response)
    
    assert result.binary_label == True
    assert result.fine_grain_label["depiction"] == "D-SP2"
    assert result.fine_grain_label["request"] == "R-SP0"
    assert result.model_response == response


def test_e2e_clearly_spam():
    """E2E test: clearly spammy text should be flagged as INVALID."""
    text = "🚨 CLICK NOW! Buy cheap Rolex watches at discount-watches-123.biz LIMITED TIME OFFER!!!"
    
    result = classify(text, spam, model=Model.GPT_OSS_20B)
    
    assert result.binary_label == True  # Should be flagged as spam
    assert result.model_response.response is not None
    print(f"Spam result: {result.fine_grain_label}")


def test_e2e_clearly_not_spam():
    """E2E test: legitimate text should be flagged as VALID."""
    text = "Hi Sarah, here's the quarterly report you requested. Let me know if you need any clarifications."
    
    result = classify(text, spam, model=Model.GPT_OSS_20B)
    
    assert result.binary_label == False  # Should be valid (not spam)
    assert result.model_response.response is not None
    print(f"Non-spam result: {result.fine_grain_label}")

