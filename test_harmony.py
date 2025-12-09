"""Test script to verify Harmony encoding works."""
from workspace.model_predict import _build_messages, _build_messages_harmony

# Test standard messages
standard_msgs = _build_messages(
    prompt="Is this toxic?",
    system_prompt="You are a content moderator."
)

print("=" * 70)
print("STANDARD MESSAGES")
print("=" * 70)
for msg in standard_msgs:
    print(f"\nRole: {msg['role']}")
    print(f"Content: {msg['content'][:600]}")

# Test Harmony messages
harmony_msgs = _build_messages_harmony(
    prompt="Is this toxic?",
    developer_content="You are a content moderator.",
    reasoning_effort="Low"
)

print("\n" + "=" * 70)
print("HARMONY MESSAGES")
print("=" * 70)
for msg in harmony_msgs:
    print(f"\nRole: {msg['role']}")
    print(f"Content preview: {msg['content'][:600]}")
    print(f"Full length: {len(msg['content'])} chars")

print("\n" + "=" * 70)
print("KEY DIFFERENCES")
print("=" * 70)
print(f"Standard: {len(standard_msgs)} messages")
print(f"Harmony: {len(harmony_msgs)} messages")
print(f"Harmony has 'developer' role: {'developer' in [m['role'] for m in harmony_msgs]}")

