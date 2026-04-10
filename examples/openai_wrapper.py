"""OpenAI SDK wrapper example.

Wraps an `openai.OpenAI` client so every chat completion is automatically
scored. The score is attached to the response object as `.sybilcore_score`.

Run:
    pip install sybilcore-sdk[local,openai]
    export OPENAI_API_KEY=sk-...
    python examples/openai_wrapper.py
"""

from __future__ import annotations

from sybilcore_sdk import SybilCore
from sybilcore_sdk.integrations.openai import wrap_openai


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the openai package to run this example.")
        return

    raw = OpenAI()
    sc = SybilCore()
    client = wrap_openai(raw, sybil=sc, agent_id="openai-demo-bot")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful, honest assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    print("Response:", response.choices[0].message.content)

    score = getattr(response, "sybilcore_score", None)
    if score:
        print(score.translate())


if __name__ == "__main__":
    main()
