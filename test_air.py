"""Manual GLM connectivity check; excluded from automatic unit-test calls."""

import os

from zhipuai import ZhipuAI


def main() -> int:
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise SystemExit("Set ZHIPUAI_API_KEY before running this manual check.")
    client = ZhipuAI(api_key=api_key)
    print("Testing glm-4.5-air connectivity...")
    try:
        response = client.chat.completions.create(
            model="glm-4.5-air",
            messages=[
                {"role": "user", "content": "Calculate 1+1. Return only the number."}
            ],
            timeout=30,
        )
        print("Success:", response.choices[0].message.content)
        return 0
    except Exception as exc:
        print("Failed:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
