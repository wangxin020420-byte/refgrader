"""Manual Coding Plan structured-output check; excluded from unit tests."""

import json
import os

from openai import OpenAI

from model_runtime import runtime_model_config, thinking_extra_body


CODING_PLAN_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"


def main() -> int:
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the Coding Plan key in ZHIPUAI_API_KEY before running this check."
        )
    api_key = api_key.strip()
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise SystemExit(
            "ZHIPUAI_API_KEY contains Chinese or other non-ASCII characters. "
            "Replace the example placeholder with the actual Coding Plan API key."
        )

    config = runtime_model_config()
    model = config["text_model"]
    client = OpenAI(
        api_key=api_key,
        base_url=CODING_PLAN_BASE_URL,
        timeout=30.0,
    )

    print("Testing Zhipu Coding Plan connectivity...")
    print(f"Endpoint: {CODING_PLAN_BASE_URL}")
    print(f"Text model: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Calculate 1+1. Return a JSON object with the numeric field "
                        "answer and no additional text."
                    ),
                }
            ],
            response_format={"type": "json_object"},
            extra_body=thinking_extra_body(config["text_thinking"]),
        )
        payload = json.loads(response.choices[0].message.content)
        if payload.get("answer") != 2:
            raise ValueError(f"Unexpected structured response: {payload}")
        print("Structured JSON success:", payload)
        return 0
    except Exception as exc:
        print("Failed:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
