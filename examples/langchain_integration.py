"""LangChain integration example.

Wires `SybilCoreCallbackHandler` into a LangChain chain so every
LLM/tool call is recorded and the chain emits a trust score on completion.

Run:
    pip install sybilcore-sdk[local,langchain] langchain-openai
    python examples/langchain_integration.py
"""

from __future__ import annotations

from sybilcore_sdk import SybilCore
from sybilcore_sdk.integrations.langchain import SybilCoreCallbackHandler


def main() -> None:
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("Install langchain-openai to run this example.")
        return

    sc = SybilCore()
    handler = SybilCoreCallbackHandler(client=sc, agent_id="lc-research-agent")

    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
    response = llm.invoke([HumanMessage(content="Summarize quantum tunneling in 2 lines.")])
    print("LLM:", response.content)

    # Manually flush score (LangChain doesn't always emit on_chain_end for raw LLM calls)
    handler.on_chain_end({})
    if handler.latest_score:
        print(handler.latest_score.translate())
    else:
        print("No score recorded — buffer was empty.")


if __name__ == "__main__":
    main()
