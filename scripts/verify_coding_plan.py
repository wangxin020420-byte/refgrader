"""Offline URL verification. Uses fake keys and MockTransport, never a live API.

Passing proves local request routing only, not server-side subscription billing.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
EXPECTED_REQUEST_URL = EXPECTED_BASE_URL + "/chat/completions"
FAKE_KEY = "offline-placeholder-not-a-real-key"
SOURCE_FILES = (
    "step0_extract_ground_truth.py", "step3_rrd_generator.py",
    "step4_vlm_grader.py", "scripts/benchmarks/project_sasbench_official_protocol.py",
)


def validate_base_url(value):
    # Strict string comparison also rejects userinfo, query, ports and lookalikes.
    if str(value) not in (EXPECTED_BASE_URL, EXPECTED_BASE_URL + "/"):
        raise ValueError("Zhipu endpoint is not the required Coding Plan URL")


def audit_sources(root=ROOT):
    """Account for every client constructor in the selected active API modules."""
    report = []
    counts = (1, 1, 4, 1)
    for filename, expected in zip(SOURCE_FILES, counts):
        tree = ast.parse((root / filename).read_text(encoding="utf-8-sig"))
        clients = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id in {"OpenAI", "ZhipuAI"}]
        if len(clients) != expected:
            raise ValueError(f"API constructor coverage changed: {filename}; inspect new call sites")
        for node in clients:
            if node.func.id != "OpenAI" or not any(k.arg == "base_url" for k in node.keywords):
                raise ValueError(f"Implicit/default endpoint is forbidden: {filename}:{node.lineno}")
        report.append({"file": filename, "explicit_client_constructors": len(clients)})
    return report


def verify():
    sources = audit_sources()
    requests, clients = [], []
    label = "initialization"

    def forbidden_network(*args, **kwargs):
        raise RuntimeError("Real network connections are forbidden in offline verification")

    def respond(request):
        if str(request.url) != EXPECTED_REQUEST_URL or request.method != "POST":
            raise ValueError("Unexpected final request URL or HTTP method")
        body = json.loads(request.content)
        requests.append({"entry": label, "url": str(request.url), "model": body["model"]})
        return httpx.Response(200, json={
            "id": "offline-mock", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "OK"}}],
        })

    def factory(*, api_key=None, base_url=None, **kwargs):
        validate_base_url(base_url)
        # Never reuse a real credential or a proxy, even if the caller supplied one.
        client = OpenAI(api_key=FAKE_KEY, base_url=base_url, max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond),
                                                 trust_env=False, follow_redirects=False))
        clients.append(client)
        return client

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", forbidden_network))
            stack.enter_context(patch("socket.create_connection", forbidden_network))
            # Patch before importing so even module-level SDK clients use fake keys.
            stack.enter_context(patch("openai.OpenAI", factory))
            modules = [importlib.import_module(name) for name in (
                "step0_extract_ground_truth", "step3_rrd_generator", "step4_vlm_grader",
                "scripts.benchmarks.project_sasbench_official_protocol")]
            teacher, rubric, grader, benchmark = modules
            for module in modules:
                stack.enter_context(patch.object(module, "OpenAI", factory))
            for module, key_name in ((teacher, "API_KEY"), (rubric, "CODING_PLAN_API_KEY"),
                                     (grader, "CODING_PLAN_API_KEY"), (grader, "GLM_API_KEY")):
                stack.enter_context(patch.object(module, key_name, FAKE_KEY))

            for module, entry in ((teacher, "teacher_extract"), (rubric, "rubric")):
                validate_base_url(module.CODING_PLAN_BASE_URL)
                client = factory(base_url=module.CODING_PLAN_BASE_URL)
                stack.enter_context(patch.object(module, "client", client))
                for model in ([teacher.VLM_MODEL_NAME] if module is teacher
                              else ["glm-5.2", rubric.VLM_MODEL_NAME]):
                    label = entry
                    module.require_client().chat.completions.create(
                        model=model, messages=[{"role": "user", "content": "offline"}])

            validate_base_url(grader.CODING_PLAN_BASE_URL)
            validate_base_url(grader.GLM_BASE_URL)
            stack.enter_context(patch.object(grader, "TEXT_MODEL_FAMILY", "glm"))
            stack.enter_context(patch.object(grader, "TEXT_MODEL_PROVIDER", "glm5"))
            stack.enter_context(patch.object(grader, "TEXT_MODEL_NAME", "glm-5.2"))
            label = "text_scoring"
            grader.call_text_model([{"role": "user", "content": "offline"}])

            # Evaluate each actual VLM client-construction expression, not a copy.
            tree = ast.parse((ROOT / "step4_vlm_grader.py").read_text(encoding="utf-8-sig"))
            vision_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name) and node.func.id == "OpenAI"
                            and any(k.arg == "base_url" and isinstance(k.value, ast.Name)
                                    and k.value.id == "GLM_BASE_URL" for k in node.keywords)]
            if len(vision_nodes) != 3:
                raise ValueError("Vision constructor coverage changed; manual review required")
            for node in vision_nodes:
                label = f"vision_constructor_line_{node.lineno}"
                client = eval(compile(ast.Expression(body=node), "<offline-client>", "eval"), vars(grader))
                client.chat.completions.create(model=grader.VLM_MODEL_NAME,
                    messages=[{"role": "user", "content": [{"type": "text", "text": "offline"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])

            with patch.dict(os.environ, {"ZHIPUAI_API_KEY": FAKE_KEY}):
                contract = benchmark._api_contract(SimpleNamespace(
                    text_provider="glm5", thinking_mode="disabled", temperature=0))
            validate_base_url(contract["base_url"])
            label = "sasbench_projection"
            benchmark.call_projection_model("offline", contract, 10)
    finally:
        for client in clients:
            client.close()
    return {
        "offline_url_verification": "passed", "expected_base_url": EXPECTED_BASE_URL,
        "source_audit": sources, "mock_requests": requests,
        "real_api_calls_made": 0, "real_credentials_used": False,
        "endpoint_requirement_satisfied": True,
        "billing_verification_required": False,
        "billing_status": "User confirmed the balance deductions came from another service, not this experiment; billing is not assessed by this offline tool.",
        "next_step": "The endpoint requirement is satisfied. This check does not launch or resume an experiment.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
