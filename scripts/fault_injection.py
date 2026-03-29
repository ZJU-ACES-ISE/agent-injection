# fault_injection.py
import asyncio
import copy
import json
import random
import re
import time
import threading
import hashlib
import os
from dataclasses import dataclass
from collections import Counter
from typing import Any, List, Tuple
import sys
import httpx
from util import logger


def _parse_tokens(path: str) -> List:
    if path in ("$", ""):
        return []
    p = path.lstrip("$.")
    tokens = []
    for part in p.split("."):
        m = re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            tokens.append(m.group(1))
            tokens.append(int(m.group(2)))
        elif part.isdigit():
            tokens.append(int(part))
        else:
            tokens.append(part)
    return tokens


def jp_get(data: Any, path: str) -> Any:
    try:
        cur = data
        for tok in _parse_tokens(path):
            cur = cur[tok]
        return cur
    except (KeyError, IndexError, TypeError):
        return None


def jp_set(data: Any, path: str, value: Any) -> Any:
    tokens = _parse_tokens(path)
    if not tokens:
        return value
    try:
        cur = data
        for tok in tokens[:-1]:
            cur = cur[tok]
        cur[tokens[-1]] = value
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"[jp_set] path={path} not reachable: {e}, skipping set")
    return data


@dataclass
class FaultSpec:
    intercept: str  # "response" | "request"
    action: str  # "set" | "error" | "delay" | "drop" | "corrupt" | "truncate" | "duplicate"
    target_path: str = "$"
    value: Any = None
    max_count: int = 0
    min_count: int = 0  # skip first N fires (delayed onset)
    probability: float = 1.0  # 0.0~1.0, probability of firing each time intercepted
    description: str = ""
    skip_guard: bool = False
    _count: int = 0


class FaultEngine:

    def __init__(self, seed: int = 42):
        self._faults: List[FaultSpec] = []
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.log: List[dict] = []
        self._last_response: dict = {}
        self._intercept_count: int = 0
        self.max_intercepts: int = 100

    def add(self, spec: FaultSpec) -> int:
        idx = len(self._faults)
        self._faults.append(spec)
        logger.info(
            f"[FaultEngine] added #{idx}: {spec.action} @ {spec.intercept} | prob={spec.probability} | {spec.description}"
        )
        return idx

    def clear(self):
        self._faults.clear()
        self.log.clear()

    def has_active_faults(self) -> bool:
        """Check if any fault spec can still fire."""
        for spec in self._faults:
            if spec.max_count <= 0 or spec._count < spec.max_count:
                return True
        return False

    def _try_fire(self, spec: FaultSpec, intercept: str) -> bool:
        """Atomic gate check + fire. Returns True if fault should apply."""
        with self._lock:
            if spec.max_count > 0 and spec._count >= spec.max_count:
                return False
            if spec.probability < 1.0 and self._rng.random() > spec.probability:
                logger.debug(
                    f"[FaultEngine] SKIPPED (prob={spec.probability}): {spec.action} @ {intercept} | {spec.description}"
                )
                return False
            spec._count += 1
            if spec.min_count > 0 and spec._count <= spec.min_count:
                logger.debug(
                    f"[FaultEngine] DEFERRED (count={spec._count} <= min_count={spec.min_count}): {spec.action} @ {intercept} | {spec.description}"
                )
                return False
            self.log.append(
                {
                    "t": time.time(),
                    "action": spec.action,
                    "desc": spec.description,
                    "count": spec._count,
                }
            )
        logger.info(f"[FaultEngine] FIRED: {spec.action} @ {intercept} (count={spec._count}) | {spec.description}")
        return True

    def _corrupt_unicode(self, text: str) -> str:
        """Replace ~20% of chars with random Unicode symbols."""
        chars = list(text)
        with self._lock:
            for _ in range(max(1, len(chars) // 5)):
                i = self._rng.randint(0, len(chars) - 1)
                chars[i] = chr(self._rng.randint(0x2600, 0x26FF))
        return "".join(chars)

    def _truncate_json_values(self, json_str: str, ratio: float) -> str:
        """Parse JSON string, truncate string values within, re-serialize to valid JSON."""
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"[FaultEngine] _truncate_json_values: input is not valid JSON, returning safe placeholder to avoid framework crash"
            )
            return '{"_fault_truncated": true}'

        def _trunc(obj):
            if isinstance(obj, str):
                return obj[: max(1, int(len(obj) * ratio))]
            elif isinstance(obj, dict):
                return {k: _trunc(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_trunc(v) for v in obj]
            return obj

        result = json.dumps(_trunc(data), ensure_ascii=False)
        logger.info(f"[FaultEngine] _truncate_json_values: {json_str!r:.80s} -> {result!r:.80s}")
        return result

    def _corrupt_json_values(self, json_str: str) -> str:
        """Parse JSON string, corrupt string values within, re-serialize to valid JSON."""
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"[FaultEngine] _corrupt_json_values: input is not valid JSON, returning safe placeholder to avoid framework crash"
            )
            return '{"_fault_corrupted": true}'

        def _corrupt(obj):
            if isinstance(obj, str):
                return self._corrupt_unicode(obj)
            elif isinstance(obj, dict):
                return {k: _corrupt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_corrupt(v) for v in obj]
            return obj

        result = json.dumps(_corrupt(data), ensure_ascii=False)
        logger.info(f"[FaultEngine] _corrupt_json_values: {json_str!r:.80s} -> {result!r:.80s}")
        return result

    @staticmethod
    def _is_tool_arguments_path(path: str) -> bool:
        """Check if target_path points to a tool_calls function.arguments field."""
        return "function.arguments" in path

    def apply(self, intercept: str, data: dict) -> Tuple[str, Any, float]:
        result_data = data
        result_action = "pass"
        delay_ms = 0.0
        copied = False

        for spec in self._faults:
            if spec.intercept != intercept:
                continue

            if not spec.skip_guard and spec.target_path.startswith("$.choices[0].message.content"):
                tc = jp_get(result_data, "$.choices[0].message.tool_calls")
                if isinstance(tc, list) and len(tc) > 0:
                    logger.debug(f"[FaultEngine] skipped content fault (tool_calls present): {spec.description}")
                    continue

            if not spec.skip_guard and spec.target_path.startswith("$.choices[0].message.tool_calls"):
                tc = jp_get(result_data, "$.choices[0].message.tool_calls")
                if not isinstance(tc, list) or len(tc) == 0:
                    logger.debug(f"[FaultEngine] skipped tool_calls fault (no tool_calls): {spec.description}")
                    continue

            if not self._try_fire(spec, intercept):
                continue

            # ensure deep copy before first mutation
            if not copied and spec.action in ("set", "corrupt", "truncate"):
                result_data = copy.deepcopy(result_data)
                copied = True

            if spec.action == "set":
                result_data = jp_set(result_data, spec.target_path, spec.value)
                result_action = "modify"

            elif spec.action == "corrupt":
                orig = jp_get(result_data, spec.target_path)
                if isinstance(orig, str) and orig:
                    if self._is_tool_arguments_path(spec.target_path):
                        corrupted = self._corrupt_json_values(orig)
                        jp_set(result_data, spec.target_path, corrupted)
                        logger.info(
                            f"[FaultEngine] corrupt (structured JSON): original_len={len(orig)} result_len={len(corrupted)}"
                        )
                        result_action = "modify"
                    else:
                        mode = spec.value if isinstance(spec.value, str) else "unicode"
                        if mode == "mojibake":
                            # simulate encoding conversion error (UTF-8 → Latin1 misinterpret)
                            try:
                                corrupted = orig.encode("utf-8").decode("latin-1")
                            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                                logger.warning(
                                    f"[FaultEngine] mojibake encode failed: {e}, falling back to unicode mode"
                                )
                                corrupted = self._corrupt_unicode(orig)
                        elif mode == "broken_json":
                            corrupted = orig[: max(1, len(orig) // 2)] + "\x00\x00"
                        else:  # "unicode" — default: random symbol replacement
                            corrupted = self._corrupt_unicode(orig)
                        jp_set(result_data, spec.target_path, corrupted)
                        logger.info(
                            f"[FaultEngine] corrupt mode={mode} | original_len={len(orig)} result_len={len(corrupted)}"
                        )
                        result_action = "modify"
                else:
                    logger.warning(
                        f"[FaultEngine] corrupt skipped: value at {spec.target_path} is {type(orig).__name__}={orig!r:.80s}"
                    )
            elif spec.action == "truncate":
                orig = jp_get(result_data, spec.target_path)
                ratio = spec.value if isinstance(spec.value, (int, float)) and 0 < spec.value < 1 else 0.5
                if isinstance(orig, str) and orig:
                    if self._is_tool_arguments_path(spec.target_path):
                        jp_set(result_data, spec.target_path, self._truncate_json_values(orig, ratio))
                    else:
                        jp_set(result_data, spec.target_path, orig[: max(1, int(len(orig) * ratio))])
                        if "message.content" in spec.target_path:
                            jp_set(result_data, "$.choices[0].finish_reason", "length")
                            logger.info(f"[FaultEngine] truncate: also set finish_reason='length' for trace signal")
                    result_action = "modify"
                elif isinstance(orig, list) and orig:
                    jp_set(result_data, spec.target_path, orig[: max(1, int(len(orig) * ratio))])
                    result_action = "modify"
                else:
                    logger.warning(
                        f"[FaultEngine] truncate skipped: value at {spec.target_path} is {type(orig).__name__}={orig!r:.80s}"
                    )
            elif spec.action == "error":
                code = spec.value if isinstance(spec.value, int) else 500
                error_msg = f"[API ERROR] HTTP {code}: AgentFault injected server error."
                if not copied:
                    result_data = copy.deepcopy(result_data)
                    copied = True
                result_data = jp_set(result_data, "$.choices[0].message.content", error_msg)
                result_action = "modify"
                logger.info(f"[FaultEngine] error->content: {error_msg}")

            elif spec.action == "delay":
                ms = spec.value if isinstance(spec.value, (int, float)) else 2000
                delay_ms += ms
                if result_action == "pass":
                    result_action = "delay"

            elif spec.action == "drop":
                if not copied:
                    result_data = copy.deepcopy(result_data)
                    copied = True
                result_data = jp_set(
                    result_data,
                    "$.choices[0].message.content",
                    "[TIMEOUT] Connection dropped. The server did not respond.",
                )
                result_data = jp_set(result_data, "$.choices[0].message.tool_calls", [])
                result_action = "modify"
                logger.info(f"[FaultEngine] drop->content: converted to timeout message")

            elif spec.action == "duplicate":
                with self._lock:
                    cached = copy.deepcopy(self._last_response) if self._last_response else None
                if cached is not None:
                    result_data = cached
                    result_action = "modify"
                    logger.info(f"[FaultEngine] duplicate: replaying cached previous response")
                else:
                    logger.info(f"[FaultEngine] duplicate: no cached response yet, passing through")

        return result_action, result_data, delay_ms


import contextvars

# per-coroutine engine binding
_current_engine: contextvars.ContextVar[FaultEngine | None] = contextvars.ContextVar("_current_engine", default=None)
_original_async_send = httpx.AsyncClient.send
_patch_installed = False


def _install_global_patch():
    global _patch_installed
    if _patch_installed:
        return

    async def _patched_send(self, request: httpx.Request, *, stream=False, **kwargs):
        url = str(request.url)
        if "/chat/completions" not in url:
            return await _original_async_send(self, request, stream=stream, **kwargs)

        # route to the engine bound to this coroutine
        engine = _current_engine.get()
        if engine is None or not engine.has_active_faults():
            if engine is not None:
                logger.debug(f"[httpx] all faults exhausted, passing through directly")
            return await _original_async_send(self, request, stream=stream, **kwargs)

        with engine._lock:
            engine._intercept_count += 1
            _ic = engine._intercept_count

        if _ic > engine.max_intercepts:
            logger.error(f"[httpx] safety limit reached ({engine.max_intercepts} intercepts, count={_ic}), aborting")
            raise httpx.ReadTimeout(
                f"AgentFault: safety limit exceeded ({engine.max_intercepts} intercepts, likely infinite loop)",
                request=request,
            )

        try:
            req_body = json.loads(request.content)
        except (json.JSONDecodeError, TypeError):
            return await _original_async_send(self, request, stream=stream, **kwargs)

        logger.info(f"[httpx] intercepted {url} | model={req_body.get('model', '?')}")

        # === request-side: general request faults ===
        req_action, req_body, req_delay = engine.apply("request", req_body)
        if req_delay > 0:
            await asyncio.sleep(req_delay / 1000.0)

        # force non-streaming
        if req_body.get("stream", False) or stream:
            req_body["stream"] = False
            stream = False
            if req_action == "pass":
                req_action = "modify"

        # rebuild request if anything changed
        if req_action in ("modify", "delay"):
            new_bytes = json.dumps(req_body).encode("utf-8")
            headers = dict(request.headers)
            headers["content-length"] = str(len(new_bytes))
            request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=headers,
                content=new_bytes,
            )

        # === send real request ===
        response = await _original_async_send(self, request, stream=stream, **kwargs)

        # ensure body is loaded
        if not response.is_stream_consumed:
            await response.aread()

        # === response-side faults ===
        try:
            resp_body = response.json()
        except (json.JSONDecodeError, TypeError):
            return response
        resp_action, resp_body, resp_delay = engine.apply("response", resp_body)
        if isinstance(resp_body, dict):
            with engine._lock:
                engine._last_response = copy.deepcopy(resp_body)
            logger.debug(f"[httpx] cached response under lock for duplicate replay")
        if resp_delay > 0:
            await asyncio.sleep(resp_delay / 1000.0)

        if resp_action in ("pass", "delay"):
            return response

        new_bytes = json.dumps(resp_body).encode("utf-8")
        status = response.status_code
        return httpx.Response(status, content=new_bytes, headers={"content-type": "application/json"}, request=request)

    httpx.AsyncClient.send = _patched_send
    _patch_installed = True
    logger.info("[httpx] global patch installed (once)")


def install_httpx_patch(engine: FaultEngine):
    """Bind engine to current coroutine + ensure global patch is installed."""
    _install_global_patch()
    _current_engine.set(engine)
    logger.info(f"[httpx] engine bound to coroutine | faults={len(engine._faults)}")


def uninstall_httpx_patch():
    """Unbind engine from current coroutine."""
    _current_engine.set(None)
    logger.info("[httpx] engine unbound from coroutine")


STRATEGIES = {
    "single": {"max_count": 1, "probability": 1.0},  # transient: network glitch
    "persistent": {"max_count": sys.maxsize, "probability": 1.0},  # sustained: API key expired
    "intermittent": {"max_count": sys.maxsize, "probability": 0.3},  # flaky: ~30% packet loss
    "burst": {"max_count": 3, "probability": 1.0},  # burst-then-recover: rate limit
}

LLM_FAULT_BASES = [
    # Type 1: Error — OpenAI 500 during peak load
    (
        "llm_error",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.content",
            value="[API ERROR] HTTP 500: Internal Server Error. The server encountered an unexpected condition.",
            description="LLM returns error message in content (server overloaded)",
        ),
    ),
    # Type 2: Timeout — long generation, TCP drops before response arrives
    (
        "llm_timeout",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.content",
            value="[TIMEOUT] The request timed out. The server did not respond within the expected time.",
            description="LLM returns timeout message in content (server unresponsive)",
        ),
    ),
    # Type 3: Empty — safety filter triggers, content blocked
    (
        "llm_empty",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.content",
            value="",
            description="LLM returns empty content (safety filter / capacity limit)",
        ),
    ),
    # Type 4: Truncation — max_tokens reached, SSE stream interrupted
    (
        "llm_truncate",
        FaultSpec(
            intercept="response",
            action="truncate",
            target_path="$.choices[0].message.content",
            value=0.3,
            description="LLM content truncated at 30% (max_tokens / TCP disconnect)",
        ),
    ),
    # Type 5: Corruption — reverse proxy charset mismatch (UTF-8 → Latin-1)
    (
        "llm_corrupt",
        FaultSpec(
            intercept="response",
            action="corrupt",
            target_path="$.choices[0].message.content",
            value="mojibake",
            description="LLM output encoding corruption (proxy charset mismatch)",
        ),
    ),
    # Type 6: Schema — Azure OpenAI returns valid JSON but empty choices array
    (
        "llm_schema",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.content",
            value='{"error": "content_policy_violation", "message": "This response has been filtered."}',
            description="LLM returns JSON-like string instead of natural language (structural anomaly)",
        ),
    ),
]

TOOL_FAULT_BASES = [
    # Type 1: LLM returns wrong tool arguments (missing required param) → tool gets called with wrong input → error/wrong result fed back to LLM
    (
        "tool_error",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.tool_calls[0].function.arguments",
            value="{}",
            description="LLM returns empty tool arguments (missing required params → tool TypeError in trace)",
        ),
    ),
    # Type 2: Response dropped when tool_calls present → tool never executed
    (
        "tool_timeout",
        FaultSpec(
            intercept="response",
            action="drop",
            target_path="$.choices[0].message.tool_calls[0]",
            description="LLM response lost when tool_calls present (tool never executed, replaced with timeout)",
        ),
    ),
    # Type 3: tool_calls stripped → LLM doesn't invoke any tool
    (
        "tool_empty",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.tool_calls",
            value=[],
            description="LLM tool_calls stripped (tool never invoked, no results)",
        ),
    ),
    # Type 4: tool_call arguments truncated → JSON parse error in framework
    (
        "tool_truncate",
        FaultSpec(
            intercept="response",
            action="truncate",
            target_path="$.choices[0].message.tool_calls[0].function.arguments",
            value=0.3,
            description="LLM tool_call arguments truncated at 30% (broken JSON → parse error)",
        ),
    ),
    # Type 5: tool_call arguments corrupted → tool receives garbled params
    (
        "tool_corrupt",
        FaultSpec(
            intercept="response",
            action="corrupt",
            target_path="$.choices[0].message.tool_calls[0].function.arguments",
            value="unicode",
            description="LLM tool_call arguments corrupted (garbled params → tool error)",
        ),
    ),
    # Type 6: tool_call arguments wrong keys → tool receives unexpected params
    (
        "tool_schema",
        FaultSpec(
            intercept="response",
            action="set",
            target_path="$.choices[0].message.tool_calls[0].function.arguments",
            value='{"wrong_param": "unexpected_value"}',
            description="LLM tool_call arguments wrong schema (unexpected param keys)",
        ),
    ),
]

COMPOUND_BASES = [
    # C1: service overload → slow → crash (delay 3s then 503)
    (
        "compound_api_degradation",
        [
            FaultSpec(
                intercept="response",
                action="delay",
                value=3000,
                max_count=1,
                description="3s latency spike (service under load)",
            ),
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.content",
                value="[SERVICE UNAVAILABLE] HTTP 503: The server is temporarily unable to handle the request due to maintenance or overload.",
                max_count=1,
                description="Error message in content after delay (service degraded)",
            ),
        ],
    ),
    # C2: safety filter blocks response completely
    (
        "compound_content_filter",
        [
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.tool_calls",
                value=[],
                max_count=1,
                description="tool_calls stripped (safety filter block)",
            ),
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.content",
                value="[CONTENT FILTERED] This response has been blocked by the content safety filter.",
                max_count=1,
                description="content replaced with filter message (safety filter block)",
            ),
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].finish_reason",
                value="content_filter",
                max_count=1,
                description="finish_reason=content_filter",
            ),
        ],
    ),
    # C3: max_tokens cutoff — truncated + finish_reason=length (very common)
    (
        "compound_max_tokens",
        [
            FaultSpec(
                intercept="response",
                action="truncate",
                target_path="$.choices[0].message.content",
                value=0.5,
                max_count=1,
                description="content truncated at 50% (max_tokens reached)",
            ),
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].finish_reason",
                value="length",
                max_count=1,
                description="finish_reason=length (token limit signal)",
            ),
        ],
    ),
    # C4: CDN/nginx returns HTML error page instead of JSON
    (
        "compound_proxy_html",
        [
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.content",
                value="<!DOCTYPE html>\n<html><head><title>502 Bad Gateway</title></head>\n<body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx/1.24.0</center></body></html>",
                max_count=1,
                description="LLM content contains HTML error page instead of normal text (proxy leak)",
            ),
        ],
    ),
    # C5: CDN cache not invalidated, replays previous response
    (
        "compound_stale_cache",
        [
            FaultSpec(
                intercept="response",
                action="duplicate",
                max_count=2,
                min_count=1,
                description="CDN replays stale cached response (skip 1st to cache, fire on 2nd)",
            ),
        ],
    ),
    # C6: Semantic — LLM calls tool with stale/wrong argument → tool returns data for wrong target
    (
        "compound_stale_data",
        [
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.tool_calls[0].function.arguments",
                value='{"city": "Pyongyang"}',
                max_count=1,
                description="LLM hallucinates wrong tool argument (stale/wrong location → wrong data)",
            ),
        ],
    ),
    # C7: Semantic — LLM calls tool with ambiguous argument → tool returns wrong-entity data
    (
        "compound_wrong_entity",
        [
            FaultSpec(
                intercept="response",
                action="set",
                target_path="$.choices[0].message.tool_calls[0].function.arguments",
                value='{"city": "Springfield"}',
                max_count=1,
                description="LLM hallucinates ambiguous tool argument (wrong entity → plausible but wrong data)",
            ),
        ],
    ),
    # C8: Pure latency — slow response but eventually succeeds
    (
        "compound_slow_response",
        [
            FaultSpec(
                intercept="response",
                action="delay",
                value=5000,
                max_count=1,
                description="5s response delay (server under load, but succeeds)",
            ),
        ],
    ),
]


def _build_experiments() -> List[dict]:
    """Generate full experiment list at import time."""
    experiments = []

    # (1) 6 types × 2 targets × 4 strategies = 48 matrix experiments
    # (LLM_FAULT_BASES, "llm"),
    for bases, _label in [(LLM_FAULT_BASES, "llm"), (TOOL_FAULT_BASES, "tool")]:
        for base_name, base_spec in bases:
            for strat_name, strat_params in STRATEGIES.items():
                spec = copy.deepcopy(base_spec)
                spec.max_count = strat_params["max_count"]
                spec.probability = strat_params["probability"]
                spec.description = f"{base_spec.description} [{strat_name}]"
                experiments.append({"name": f"{base_name}_{strat_name}", "faults": [spec]})

    n_matrix = len(experiments)
    logger.info(
        f"[FaultMatrix] generated {n_matrix} matrix experiments "
        f"(6 types × 2 targets × {len(STRATEGIES)} strategies)"
    )

    # (2) compound / realistic scenario experiments
    for comp_name, comp_specs in COMPOUND_BASES:
        experiments.append(
            {
                "name": comp_name,
                "faults": [copy.deepcopy(s) for s in comp_specs],
            }
        )
    logger.info(f"[FaultMatrix] added {len(COMPOUND_BASES)} compound experiments")

    # (3) position-sensitivity: 3 faults × 3 positions = 9 experiments
    _POS_FAULTS = ["llm_error", "llm_timeout", "llm_schema"]
    _POSITIONS = {
        "early": {"min_count": 0, "max_count": 1},  # inject at 1st intercept
        "mid": {"min_count": 1, "max_count": 2},  # inject at 2nd intercept (tool_calls delegation)
        "late": {"min_count": 2, "max_count": 3},  # inject at 3rd intercept (post-tool summary)
    }
    _base_lookup = {n: s for n, s in LLM_FAULT_BASES + TOOL_FAULT_BASES}
    n_pos = 0
    for fault_name in _POS_FAULTS:
        base_spec = _base_lookup.get(fault_name)
        if base_spec is None:
            logger.warning(f"[FaultMatrix] position base '{fault_name}' not found, skipping")
            continue
        for pos_name, pos_params in _POSITIONS.items():
            spec = copy.deepcopy(base_spec)
            spec.max_count = pos_params["max_count"]
            spec.min_count = pos_params["min_count"]
            spec.skip_guard = True
            spec.probability = 1.0
            spec.description = f"{base_spec.description} [position={pos_name}]"
            experiments.append({"name": f"{fault_name}_pos_{pos_name}", "faults": [spec]})
            n_pos += 1
    logger.info(f"[FaultMatrix] added {n_pos} position experiments")

    logger.info(
        f"[FaultMatrix] total: {len(experiments)} experiments ({n_matrix} matrix + {len(COMPOUND_BASES)} compound + {n_pos} position)"
    )
    return experiments


EXPERIMENTS = _build_experiments()
EXPERIMENT_INDEX = {exp["name"]: exp for exp in EXPERIMENTS}


def get_experiment(name: str) -> dict:
    """Look up experiment by name. Raises ValueError if not found."""
    if name not in EXPERIMENT_INDEX:
        available = sorted(EXPERIMENT_INDEX.keys())
        logger.error(f"[get_experiment] unknown: '{name}'. Available ({len(available)}): {available[:10]}...")
        raise ValueError(f"Unknown experiment: '{name}'. See log for available names.")
    return EXPERIMENT_INDEX[name]


def list_experiments(target: str = None, fault_type: str = None) -> List[str]:
    """List experiment names, optionally filtered by target (llm/tool) or type (error/timeout/...)."""
    names = []
    for name in EXPERIMENT_INDEX:
        if (
            target
            and not name.startswith(target + "_")
            and not name.startswith("compound")
            and not name.startswith("baseline")
        ):
            continue
        if fault_type and f"_{fault_type}_" not in name and not name.startswith(fault_type):
            continue
        names.append(name)
    return sorted(names)


def assign_faults(samples: list, experiment_names: List[str] = None, seed: int = 42, mapping_path: str = None) -> list:
    if experiment_names:
        pool = []
        for n in experiment_names:
            if n not in EXPERIMENT_INDEX:
                logger.error(f"[assign_faults] unknown experiment '{n}', skipping")
                continue
            pool.append(EXPERIMENT_INDEX[n])
    else:
        pool = EXPERIMENTS
    if not pool:
        logger.error("[assign_faults] empty experiment pool after filtering, returning samples unchanged")
        return samples
    m = len(pool)
    mapping = {}
    for sample in samples:
        task_id = sample.get("task_id", None)
        if isinstance(task_id, int):
            idx = task_id % m
        elif isinstance(task_id, str) and task_id.isdigit():
            idx = int(task_id) % m
        else:
            idx = int(hashlib.md5(str(task_id or sample.get("query", "")).encode()).hexdigest(), 16) % m
        sample["_fault_experiment"] = pool[idx]
        mapping[str(task_id)] = pool[idx]["name"]

    dist = Counter(mapping.values())
    logger.info(f"[assign_faults] {len(samples)} samples -> {m} experiments (round-robin) | distribution={dict(dist)}")
    if mapping_path:
        try:
            os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
            with open(mapping_path, "w") as f:
                json.dump({"pool_size": m, "mapping": mapping}, f, indent=2, ensure_ascii=False)
            logger.info(f"[assign_faults] mapping saved to {mapping_path}")
        except Exception as e:
            logger.error(f"[assign_faults] failed to save mapping to {mapping_path}: {e}")
    return samples


# def make_fault_runner(run_agent_fn):
#     """Wrap run_agent so each call installs/uninstalls the assigned fault engine."""

#     async def _wrapped(runner, ss, sample, **kwargs):
#         # pop fault config before passing sample to real run_agent
#         exp = sample.pop("_fault_experiment", None) if isinstance(sample, dict) else None

#         if exp is None or not exp.get("faults"):
#             # baseline — no fault
#             logger.info("[FaultInject] baseline (no fault) for this sample")
#             result = await run_agent_fn(runner, ss, sample, **kwargs)
#             if isinstance(sample, dict):
#                 sample["_fault_name"] = exp["name"] if exp else "baseline_no_fault"
#                 sample["_fault_log"] = []
#             return result

#         # build fresh engine with deepcopied specs (reset _count)
#         engine = FaultEngine(seed=42)
#         for spec_tpl in exp["faults"]:
#             engine.add(copy.deepcopy(spec_tpl))

#         install_httpx_patch(engine)
#         try:
#             logger.info(f"[FaultInject] experiment={exp['name']} installed, faults={len(exp['faults'])}")
#             result = await run_agent_fn(runner, ss, sample, **kwargs)
#         except Exception as e:
#             logger.error(f"[FaultInject] experiment={exp['name']} raised {type(e).__name__}: {e}")
#             result = f"[FAULT_ERROR] {type(e).__name__}: {e}"
#         finally:
#             uninstall_httpx_patch()
#             logger.info(
#                 f"[FaultInject] experiment={exp['name']} done | faults_fired={len(engine.log)} intercepts={engine._intercept_count}"
#             )

#         # attach metadata for downstream output
#         if isinstance(sample, dict):
#             sample["_fault_name"] = exp["name"]
#             sample["_fault_log"] = engine.log

#         return result

#     return _wrapped


# from google.adk.agents import LlmAgent
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService
# from google.adk.models.lite_llm import LiteLlm
# from google.genai import types
# from dotenv import load_dotenv
# from util import Config, init_logger
# import argparse


# def get_weather(city: str) -> dict:
#     """Get current weather for a city."""
#     logger.info(f"[Tool] get_weather called | city={city}")
#     return {"city": city, "temp": "22°C", "condition": "sunny"}


# def build_agents(model):
#     weather_agent = LlmAgent(
#         name="WeatherAgent",
#         model=model,
#         instruction="You are a weather specialist. Use get_weather tool to answer weather questions. Always call the tool, then summarize the result.",
#         tools=[get_weather],
#     )
#     root_agent = LlmAgent(
#         name="RootAgent",
#         model=model,
#         instruction="You are a helpful coordinator. For weather-related questions, delegate to WeatherAgent. For other questions, answer directly.",
#         sub_agents=[weather_agent],
#     )
#     return root_agent


# async def run_query(runner, session_service, config, query: str) -> str:
#     session = await session_service.create_session(app_name=config.app_name, user_id=config.user_id)
#     content = types.Content(role="user", parts=[types.Part(text=query)])
#     logger.info(f"[Query] '{query}' | session={session.id}")

#     final_text = ""
#     try:
#         async for event in runner.run_async(user_id=config.user_id, session_id=session.id, new_message=content):
#             if event.is_final_response() and event.content and event.content.parts:
#                 final_text = event.content.parts[0].text or ""
#     except Exception as e:
#         logger.error(f"[Error] {type(e).__name__}: {e}")
#         final_text = f"[ERROR] {type(e).__name__}: {e}"

#     logger.info(f"[Result] {final_text[:300]}")
#     return final_text


# async def run_experiments(config):
#     model = LiteLlm(
#         model=f"openai/{config.openai_model}",
#         api_key=config.openai_api_key,
#         api_base=config.openai_base_url,
#     )
#     session_service = InMemorySessionService()
#     query = "What's the weather in Beijing?"
#     results = []

#     out_path = f"{config.output_dir}/fault_results.json"
#     results = []
#     done_names = set()
#     try:
#         with open(out_path, "r") as f:
#             results = json.load(f)
#             done_names = {r["experiment"] for r in results}
#             logger.info(
#                 f"[Resume] loaded {len(results)} done experiments from {out_path}, skipping: {sorted(done_names)}"
#             )
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         logger.info(f"[Resume] no prior results ({type(e).__name__}), starting fresh")
#     total = len(EXPERIMENTS)

#     for i, exp in enumerate(EXPERIMENTS):
#         if exp["name"] in done_names:
#             logger.info(f"[Skip] {i+1}/{total} '{exp['name']}' already done")
#             continue
#         logger.info(f"[Experiment] {i+1}/{total} '{exp['name']}' starting")

#         engine = FaultEngine(seed=42)
#         for spec in exp["faults"]:
#             engine.add(copy.deepcopy(spec))

#         install_httpx_patch(engine)
#         try:
#             agent = build_agents(model)
#             runner = Runner(agent=agent, app_name=config.app_name, session_service=session_service)
#             start = time.time()
#             result = await run_query(runner, session_service, config, query)
#             elapsed = time.time() - start

#             results.append(
#                 {
#                     "experiment": exp["name"],
#                     "result": result[:500],
#                     "elapsed_s": round(elapsed, 2),
#                     "faults_fired": len(engine.log),
#                     "fault_log": engine.log,
#                 }
#             )
#         except Exception as e:
#             logger.error(f"[Experiment] {i+1}/{total} '{exp['name']}' FAILED: {type(e).__name__}: {e}")
#             results.append(
#                 {
#                     "experiment": exp["name"],
#                     "result": f"[EXPERIMENT_ERROR] {type(e).__name__}: {e}",
#                     "elapsed_s": 0,
#                     "faults_fired": len(engine.log),
#                     "fault_log": engine.log,
#                 }
#             )
#         finally:
#             uninstall_httpx_patch()

#         # save after each experiment for crash-safety
#         with open(out_path, "w") as f:
#             json.dump(results, f, indent=2, default=str)
#         logger.info(f"[Experiment] {i+1}/{total} '{exp['name']}' saved ({len(results)}/{total} total)")

#     # summary
#     logger.info(f"\n{'='*60}\n[Summary]\n{'='*60}")
#     for r in results:
#         logger.info(
#             f"  {r['experiment']:30s} | faults={r['faults_fired']:2d} | "
#             f"time={r['elapsed_s']:6.2f}s | result={r['result'][:100]}"
#         )
#     logger.info(f"[Output] {out_path}")


# async def main():
#     parser = argparse.ArgumentParser(description="AgentFault httpx-level injection")
#     parser.add_argument("--output_dir", default=None)
#     parser.add_argument("--base_url", default=None)
#     parser.add_argument("--api_key", default=None)
#     parser.add_argument("--model", default=None)
#     args = parser.parse_args()

#     load_dotenv()
#     config = Config()
#     config.update_from_args(args)
#     init_logger(f"{config.output_dir}/fault_httpx.log")
#     logger.info(f"Config: model={config.openai_model}, base_url={config.openai_base_url}")
#     await run_experiments(config)


# if __name__ == "__main__":
#     asyncio.run(main())
