# util.py
import json
import os
import base64
import threading
import atexit
import logging
from datetime import datetime
from unittest.mock import MagicMock
import requests
import opentelemetry.proto.trace.v1.trace_pb2 as trace_pb2
from google.protobuf.json_format import MessageToDict


class CustomHTTPInterceptor:
    """Intercept HTTP requests to observability platforms and save trace data locally."""

    _ID_FIELDS = ("trace_id", "span_id", "parent_span_id")
    _file_counter = 0
    _file_counter_lock = threading.Lock()

    def __init__(self, output_dir="./traces", intercept_keywords=None, otlp_keywords=None, logger=None):
        self.output_dir = output_dir
        self.intercept_keywords = set(intercept_keywords or ["agentops.ai", "smith.langchain", "langfuse", "6006"])
        self.otlp_keywords = set(otlp_keywords or ["otlp.agentops.ai", "langfuse", "6006"])
        self._logger = logger or logging.getLogger(__name__)
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._pending_zero = threading.Event()
        self._pending_zero.set()
        self._setup_patches()
        atexit.register(lambda: self._pending_zero.wait(timeout=30))

    def _should_intercept(self, url):
        return any(k in str(url).lower() for k in self.intercept_keywords)

    def _create_mock_response(self):
        m = MagicMock()
        m.status_code, m.json.return_value, m.raise_for_status = 200, {"success": True}, lambda: None
        return m

    def _parse_otlp_data(self, data: bytes):
        try:
            td = trace_pb2.TracesData()
            td.ParseFromString(data)
            return self._normalize(
                MessageToDict(td, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)
            )
        except Exception as e:
            self._logger.error(f"OTLP parse error: {e}")
            return None

    def _normalize(self, data):
        if isinstance(data, list):
            return [self._normalize(i) for i in data]
        if not isinstance(data, dict):
            return data
        for f in self._ID_FIELDS:
            if f in data and data[f]:
                try:
                    data[f] = "0x" + base64.b64decode(data[f]).hex()
                except Exception:
                    pass
        if "attributes" in data and isinstance(data["attributes"], list):
            data["attributes"] = {
                a["key"]: self._extract_val(a.get("value", a))
                for a in data["attributes"]
                if isinstance(a, dict) and "key" in a
            }
        for k, v in list(data.items()):
            if isinstance(v, (dict, list)):
                data[k] = self._normalize(v)
        return data

    def _extract_val(self, v):
        if not isinstance(v, dict):
            return v
        for tk, fn in {"string_value": str, "int_value": int, "double_value": float, "bool_value": bool}.items():
            if tk in v:
                return fn(v[tk])
        if "array_value" in v and isinstance(v["array_value"], dict):
            return [self._extract_val(x) for x in v["array_value"].get("values", [])]
        return v

    def _process_data(self, data, url):
        if data is None or isinstance(data, (dict, list, int, float, bool)):
            return data
        if isinstance(data, (bytes, bytearray)):
            if any(k in str(url).lower() for k in self.otlp_keywords):
                r = self._parse_otlp_data(data)
                if r is not None:
                    return r
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                return base64.b64encode(data).decode()
        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                return data
        return data

    def _save_request(self, method, url, data=None):
        if not self.output_dir:
            return
        with self._pending_lock:
            self._pending += 1
            self._pending_zero.clear()
        threading.Thread(target=self._do_save, args=(method, url, data), daemon=False).start()

    def _do_save(self, method, url, data):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with CustomHTTPInterceptor._file_counter_lock:
                CustomHTTPInterceptor._file_counter += 1
                seq = CustomHTTPInterceptor._file_counter
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{seq}.json"
            with open(os.path.join(self.output_dir, fname), "w", encoding="utf-8") as f:
                json.dump(
                    {"method": method, "url": str(url), "data": self._process_data(data, url)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            self._logger.error(f"trace save failed: {e}")
        finally:
            with self._pending_lock:
                self._pending -= 1
                if self._pending == 0:
                    self._pending_zero.set()

    def _setup_patches(self):
        orig_req, orig_sess = requests.request, requests.Session.request
        me = self

        def p1(method, url, **kw):
            if me._should_intercept(url):
                me._save_request(method, url, kw.get("json") or kw.get("data"))
                return me._create_mock_response()
            return orig_req(method, url, **kw)

        def p2(self_, method, url, **kw):
            if me._should_intercept(url):
                me._save_request(method, url, kw.get("json") or kw.get("data"))
                return me._create_mock_response()
            return orig_sess(self_, method, url, **kw)

        requests.request, requests.Session.request = p1, p2


def setup_tracing(project="SimpleAgent", logger=None, trace_dir="./traces"):
    """Initialize OpenTelemetry tracing and HTTP interceptor."""
    os.environ.update({"OTEL_BSP_SCHEDULE_DELAY": "999999", "OTEL_BSP_MAX_QUEUE_SIZE": "200000"})
    from phoenix.otel import register

    tp = register(endpoint="http://localhost:6006", project_name=project, batch=True, auto_instrument=True)
    return tp, CustomHTTPInterceptor(output_dir=trace_dir, logger=logger)


# ============ Fault Injection Facade ============


def setup_fault_injection(experiment_name: str = None, seed: int = 42):
    if not experiment_name:
        logger.info("[fault] no experiment specified, skipping fault injection")
        return None

    import copy as _copy
    from fault_injection import get_experiment, FaultEngine, install_httpx_patch

    exp = get_experiment(experiment_name)
    engine = FaultEngine(seed=seed)
    for spec in exp["faults"]:
        engine.add(_copy.deepcopy(spec))
    install_httpx_patch(engine)
    logger.info(f"[fault] installed experiment='{experiment_name}' with {len(exp['faults'])} fault spec(s)")
    return engine


def teardown_fault_injection(engine):
    if engine is None:
        return
    from fault_injection import uninstall_httpx_patch

    uninstall_httpx_patch()
    logger.info(f"[fault] teardown | faults_fired={len(engine.log)} | intercepts={engine._intercept_count}")


def build_fault_assignment(task_ids: list, seed: int = 42) -> dict:
    """Deterministic round-robin assignment: task_id -> experiment_name."""
    from fault_injection import EXPERIMENTS

    if not EXPERIMENTS:
        logger.error("[fault_assign] EXPERIMENTS is empty")
        return {}

    m = len(EXPERIMENTS)
    sorted_ids = sorted(task_ids, key=str)
    mapping = {}
    for i, tid in enumerate(sorted_ids):
        mapping[str(tid)] = EXPERIMENTS[i % m]["name"]

    from collections import Counter

    dist = Counter(mapping.values())
    logger.info(
        f"[fault_assign] {len(task_ids)} tasks -> {m} experiments (round-robin) | "
        f"min={min(dist.values())} max={max(dist.values())}"
    )
    return mapping


import os
import json
import asyncio
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from loguru import logger
from dataclasses import dataclass, field
from openai import OpenAI, AsyncOpenAI


@dataclass
class Config:
    # path config
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "output")))

    # LLM config
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "your-api-key"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    app_name: str = "default_app"
    user_id: str = "default_user"
    session_id: str = "default_session_id"
    llm_max_concurrency: int = 5

    # embedding config
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    )
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "ollama"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:567m-fp16"))
    embedding_max_concurrency: int = 5

    # Judge config ranges
    reasons_range: Tuple[int, int] = (0, 10)
    fault_root_cause_range: Tuple[int, int] = (0, 10)
    execute_principle_range: Tuple[int, int] = (0, 10)
    judge_principle_range: Tuple[int, int] = (0, 10)

    def update_from_args(self, args):
        """Override config from command line args (only override non-None values)"""
        if hasattr(args, "output_dir") and args.output_dir:
            self.output_dir = args.output_dir
        if hasattr(args, "base_url") and args.base_url:
            self.openai_base_url = args.base_url
        if hasattr(args, "api_key") and args.api_key:
            self.openai_api_key = args.api_key
        if hasattr(args, "model") and args.model:
            self.openai_model = args.model
        if hasattr(args, "embedding_base_url") and args.embedding_base_url:
            self.embedding_base_url = args.embedding_base_url
        if hasattr(args, "embedding_api_key") and args.embedding_api_key:
            self.embedding_api_key = args.embedding_api_key
        if hasattr(args, "embedding_model") and args.embedding_model:
            self.embedding_model = args.embedding_model
        if hasattr(args, "reasons_range") and args.reasons_range:
            self.reasons_range = args.reasons_range
        if hasattr(args, "fault_root_cause_range") and args.fault_root_cause_range:
            self.fault_root_cause_range = args.fault_root_cause_range
        if hasattr(args, "execute_principle_range") and args.execute_principle_range:
            self.execute_principle_range = args.execute_principle_range
        if hasattr(args, "judge_principle_range") and args.judge_principle_range:
            self.judge_principle_range = args.judge_principle_range
        return self


# ============ init_logger ============
def init_logger(log_file: str, level: str = "INFO"):
    """Configure logger with colorized output"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        Path(log_file),
        level=level,
        rotation="1 day",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] ({name}:{function}:{line}) {message}",
    )
    return logger


# ============ log_api ============
def log_api(log_file: str, data: dict):
    """Record API call logs to jsonl file"""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        record = {"timestamp": datetime.now().isoformat(), **data}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============ acall_llm ============
class RetryableError(Exception):
    pass


# ============ LLM semaphore cache ============
_semaphore_cache: Dict[str, asyncio.Semaphore] = {}


def _get_semaphore(key: str, limit: int) -> asyncio.Semaphore:
    if key not in _semaphore_cache:
        _semaphore_cache[key] = asyncio.Semaphore(limit)
    return _semaphore_cache[key]


# ============ Embedding ============
def call_embedding(text: str, config) -> List[float]:
    """Sync single embedding call"""
    log_file = os.path.join(getattr(config, "output_dir", "logs"), "call_embedding.jsonl")
    client = OpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)
    try:
        response = client.embeddings.create(model=config.embedding_model, input=[text], encoding_format="float")
        embedding = response.data[0].embedding
        log_api(log_file, {"input": text, "embedding": embedding[:50]})
        return embedding
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        raise


async def acall_embedding(text: str, config) -> List[float]:
    """Async single embedding call"""
    async with _get_semaphore("embedding", getattr(config, "embedding_max_concurrency", 5)):
        log_file = os.path.join(getattr(config, "output_dir", "logs"), "call_embedding.jsonl")
        client = AsyncOpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)
        try:
            response = await client.embeddings.create(
                model=config.embedding_model, input=[text], encoding_format="float"
            )
            embedding = response.data[0].embedding
            log_api(log_file, {"input": text, "embedding": embedding[:50]})
            return embedding
        except Exception as e:
            logger.error(f"Embedding call failed: {e}")
            raise


async def acall_embedding_batch(texts: List[str], config, batch_size: int = 100) -> List[List[float]]:
    """Batch embedding call - process multiple texts in batches"""
    if not texts:
        return []
    all_embeddings: List[List[float]] = []
    log_file = os.path.join(getattr(config, "output_dir", "logs"), "call_embedding.jsonl")
    client = AsyncOpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        async with _get_semaphore("embedding", getattr(config, "embedding_max_concurrency", 5)):
            try:
                response = await client.embeddings.create(
                    model=config.embedding_model, input=batch_texts, encoding_format="float"
                )
                embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(embeddings)
                log_api(log_file, {"input_count": len(batch_texts), "output_count": len(embeddings)})
            except Exception as e:
                logger.error(f"Batch embedding failed at batch {i // batch_size + 1}: {e}")
                raise
    logger.info(f"Batch embedding done: {len(texts)} texts -> {len(all_embeddings)} embeddings")
    return all_embeddings


async def acall_llm(
    messages: List[Dict],
    config,
    output_schema: Optional[type[BaseModel]] = None,
) -> str:
    """Async LLM call with JSON schema support and auto-retry"""
    # Semaphore for concurrency control

    async with _get_semaphore("llm", getattr(config, "llm_max_concurrency", 5)):
        log_file = os.path.join(getattr(config, "output_dir", "logs"), "call_llm.jsonl")
        # Build LLM client
        model_lower = config.openai_model.lower()
        extra_params = {}
        if "gpt" in model_lower:
            # none, low, medium, high. seed is ok too (minimal)
            extra_params["model_kwargs"] = {"reasoning_effort": "none"}
            pass
        elif "qwen" in model_lower or "seed" in model_lower or "glm" in model_lower:
            extra_params["temperature"] = 0.7
            # extra_params["model_kwargs"] = {"reasoning_effort": "medium"}
            extra_params["extra_body"] = {"thinking": {"type": "disabled"}}
        elif "claude" in model_lower:
            extra_params["temperature"] = 0.7

        llm = ChatOpenAI(
            model=config.openai_model,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=600,
            max_retries=0,
            **extra_params,
        )

        max_retries = 10
        for attempt in range(max_retries):
            try:
                if output_schema:
                    # Structured JSON output
                    structured_llm = llm.with_structured_output(
                        output_schema,
                        method="json_schema",
                        include_raw=True,
                    )
                    raw_response = await structured_llm.ainvoke(messages)
                    response = raw_response["raw"]
                    parsed = raw_response.get("parsed")

                    if parsed is None:
                        raw_content = getattr(response, "content", str(response))
                        logger.error(f"LLM structured output parsing failed, raw={raw_content[:500]}")
                        raise ValueError(f"Structured output parsing returned None")

                    # Log API call
                    log_api(
                        log_file,
                        {
                            "input": messages,
                            "output": response.model_dump() if hasattr(response, "model_dump") else str(response),
                        },
                    )
                    logger.debug(
                        f"LLM call success, model={config.openai_model}, tokens={getattr(response, 'usage_metadata', 'N/A')}"
                    )
                    return parsed.model_dump_json()
                else:
                    # Plain text output
                    response = await llm.ainvoke(messages)
                    log_api(
                        log_file,
                        {
                            "input": messages,
                            "output": response.model_dump() if hasattr(response, "model_dump") else str(response),
                        },
                    )
                    logger.debug(f"LLM call success, model={config.openai_model}")
                    return response.content

            except Exception as e:
                err_str = str(e).lower()
                retryable_keywords = ["timeout", "rate limit", "rate_limit", "429", "503", "overloaded", "connection"]

                if any(kw in err_str for kw in retryable_keywords):
                    if attempt < max_retries - 1:
                        wait_time = 5 * (2**attempt) + random.uniform(0, 1)
                        match = re.search(r"retry after (\d+)", err_str)
                        if match:
                            wait_time = int(match.group(1)) + 1
                        logger.warning(f"LLM retry {attempt+1}/{max_retries}, wait {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"LLM failed after {max_retries} retries: {e}")
                        raise RetryableError(str(e)) from e

                logger.error(f"LLM call failed (permanent): {e}")
                raise
