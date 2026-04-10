"""SDK integrations — adapters for popular agent frameworks.

Each integration converts framework-specific events into SDK `Event`
objects and feeds them through `SybilCore.score()`.

Optional dependencies (install with extras):
    pip install sybilcore-sdk[langchain]
    pip install sybilcore-sdk[openai]
    pip install sybilcore-sdk[anthropic]
"""
