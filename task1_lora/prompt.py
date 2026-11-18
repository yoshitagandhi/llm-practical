"""Prompt template for Task 1, shared by training and evaluation.

Both the base model and the fine-tuned model must see exactly the same
prompt, otherwise the comparison is not fair. Keeping the template in one
place guarantees that.
"""

SYSTEM_PROMPT = "You convert voice-assistant requests into JSON. Reply with JSON only."


def build_user_message(utterance, intents, slot_types):
    """Return the user turn: task rules, allowed labels and the request."""
    intent_list = ", ".join(intents)
    slot_list = ", ".join(slot_types)
    return (
        'Convert the request into JSON with exactly two keys:\n'
        '- "intent": one of the allowed intents\n'
        '- "slots": a list of {"type": ..., "value": ...} objects, where each value '
        'is copied exactly from the request. Use [] if there are no slots.\n\n'
        f"Allowed intents: {intent_list}\n"
        f"Allowed slot types: {slot_list}\n\n"
        f"Request: {utterance}"
    )


def build_messages(utterance, intents, slot_types, target=None):
    """Return chat messages for the tokenizer's chat template.

    For training, pass the gold JSON string as `target` so it becomes the
    assistant turn. For evaluation, leave `target` as None and let the model
    generate the assistant turn.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(utterance, intents, slot_types)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


if __name__ == "__main__":
    # Quick visual check of the template with made-up labels.
    example = build_messages(
        "wake me up at five am this week",
        intents=["alarm_set", "alarm_query"],
        slot_types=["date", "time"],
    )
    for message in example:
        print(f"[{message['role']}]\n{message['content']}\n")
    assert len(example) == 2
    assert "Request: wake me up at five am this week" in example[1]["content"]
