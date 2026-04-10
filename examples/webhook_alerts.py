"""Webhook alert registration example.

Demonstrates registering a webhook with a remote SybilCore API server
so high-trust agent events POST to your alerting endpoint.

Run a local server first:
    python -m sybilcore.api.run_server --port 8765
Then run this example.
"""

from __future__ import annotations

from sybilcore_sdk import SybilCore, Tier


def main() -> None:
    sc = SybilCore(api_key="local-dev", endpoint="http://localhost:8765")
    info = sc.register_webhook(
        callback_url="https://alerts.example.com/sybilcore",
        min_tier=Tier.FLAGGED,
    )
    print("Webhook registered:")
    print(info)


if __name__ == "__main__":
    main()
