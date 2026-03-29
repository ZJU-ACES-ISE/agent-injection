# parse_trace.py
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
from abc import ABC, abstractmethod
from collections import defaultdict
import re


class BaseTraceParser(ABC):
    """Base Trace Parser"""

    def __init__(self, trace: List[Dict]):
        self.trace = self._deduplicate_trace(trace)
        self.span_map = {t["span_id"]: t for t in self.trace}

    def _deduplicate_trace(self, trace: List[Dict]) -> List[Dict]:
        seen_ids = set()
        deduplicated_trace = []
        for span in trace:
            span_id = span.get("span_id")
            if span_id and span_id not in seen_ids:
                seen_ids.add(span_id)
                deduplicated_trace.append(span)
        return deduplicated_trace

    def parse_json_value(self, span: Dict, field: str) -> Any:
        """Safely parse JSON field"""
        try:
            value = span.get("attributes", {}).get(field, "{}")
            return json.loads(value) if isinstance(value, str) else value
        except:
            return {}

    @abstractmethod
    def extract_agent_steps(self) -> List[Dict]:
        pass

    @abstractmethod
    def extract_agent_settings(self) -> Dict:
        pass

    @abstractmethod
    def extract_agent_dependency(self) -> Dict:
        pass

    def parse_all(self) -> Tuple:
        agent_steps = self.extract_agent_steps()
        agent_settings = self.extract_agent_settings()
        agent_dependency = self.extract_agent_dependency()
        return agent_steps, agent_settings, agent_dependency


class ADKPhoenixParser(BaseTraceParser):
    _SCOPES = {"openinference.instrumentation.google_adk"}
    # Also accept OpenAI instrumentation (used by mini_se via LiteLlm)
    _OPENAI_SCOPES = {"openinference.instrumentation.openai", "openinference.instrumentation.litellm"}

    def __init__(self, trace: List[Dict]):
        self._all_span_map = {s["span_id"]: s for s in trace}
        # Include ADK-scoped spans as before
        adk_spans = [s for s in trace if s.get("_scope") in self._SCOPES]
        # If no ADK spans found, fall back to OpenAI-scoped LLM spans (mini_se case)
        if not adk_spans:
            adk_spans = [s for s in trace if s.get("_scope") in self._OPENAI_SCOPES]
            self._is_openai_only = True
        else:
            self._is_openai_only = False
        super().__init__(adk_spans)

    def _find_agent_name(self, span) -> str:
        """Walk up parent chain to find nearest agent_run [name] span.
        Fallback: extract agent name from system prompt 'internal name is "xxx"'."""
        if not self._is_openai_only:
            # Normal ADK path: walk up parent chain for agent_run spans
            visited = set()
            current_id = span.get("parent_span_id")
            while current_id and current_id not in visited:
                visited.add(current_id)
                parent = self._all_span_map.get(current_id)
                if parent is None:
                    break
                if "agent_run" in parent.get("name", ""):
                    match = re.search(r"agent_run \[(.+?)\]", parent.get("name", ""))
                    if match:
                        return match.group(1)
                current_id = parent.get("parent_span_id")
        # Fallback: extract agent name from system prompt 'internal name is "xxx"'
        # Search in current span and all descendants for llm.input_messages
        attrs = span.get("attributes", {})
        sys_content = attrs.get("llm.input_messages.0.message.content", "")
        if not sys_content:
            # Walk descendants (call_llm -> generate_content -> ChatCompletion)
            queue = [span.get("span_id")]
            checked = set()
            while queue and not sys_content:
                sid = queue.pop(0)
                if sid in checked:
                    continue
                checked.add(sid)
                for child in self._all_span_map.values():
                    if child.get("parent_span_id") == sid:
                        c = child.get("attributes", {}).get("llm.input_messages.0.message.content", "")
                        if c:
                            sys_content = c
                            break
                        queue.append(child["span_id"])
        match = re.search(r'internal name is ["\'](.+?)["\']', sys_content)
        if match:
            return match.group(1)
        return "unknown"

    def _get_adk_descendants(self, span_id: str) -> List[Dict]:
        """Get all ADK-scope descendants, traversing through non-ADK intermediary spans."""
        descendants = []
        queue = [span_id]
        visited = set()
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            for s in self._all_span_map.values():
                if s.get("parent_span_id") == current_id:
                    if s.get("_scope") in self._SCOPES:
                        descendants.append(s)
                    queue.append(s["span_id"])
        return descendants

    def _find_tool_response(self, tool_call_id: str):
        """Find tool response by matching gen_ai.tool.call.id across all execute_tool spans."""
        if not tool_call_id:
            return None
        for tool_span in self.trace:
            if "execute_tool" not in tool_span.get("name", ""):
                continue
            if tool_span.get("attributes", {}).get("gen_ai.tool.call.id") == tool_call_id:
                return self.parse_json_value(tool_span, "output.value")
        return None

    def _is_llm_span(self, span) -> bool:
        """Check if span is an LLM call (call_llm for ADK, ChatCompletion for OpenAI)."""
        name = span.get("name", "")
        if name.startswith("call_llm"):
            return True
        # OpenAI-only: ChatCompletion spans with LLM kind
        if self._is_openai_only and name == "ChatCompletion":
            return span.get("attributes", {}).get("openinference.span.kind") == "LLM"
        return False

    def extract_agent_steps(self) -> List[Dict]:
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        for span in sorted(self.trace, key=lambda x: x["start_time_unix_nano"]):
            if not self._is_llm_span(span):
                continue

            step_num += 1
            attrs = span.get("attributes", {})

            # Get agent name by walking up parent chain
            agent_name = self._find_agent_name(span)

            # Parse input from llm.input_messages, only keep user role messages
            user_input = []
            idx = 0
            while True:
                role = attrs.get(f"llm.input_messages.{idx}.message.role")
                if role is None:
                    break
                if role == "user":
                    content = attrs.get(f"llm.input_messages.{idx}.message.content")
                    if not content:
                        content = attrs.get(f"llm.input_messages.{idx}.message.contents.0.message_content.text", "")
                    if content and not user_input:
                        user_input.append({"role": "user", "content": content})
                idx += 1

            # Fallback: parse user input from llm_request contents
            if not user_input:
                llm_req = self.parse_json_value(span, "gcp.vertex.agent.llm_request")
                if isinstance(llm_req, dict):
                    for cont in llm_req.get("contents", []):
                        if cont.get("role") == "user":
                            for part in cont.get("parts", []):
                                if "text" in part and not user_input:
                                    user_input.append({"role": "user", "content": part["text"]})

            # Parse output text: try llm_response first, then output.value
            output_text = ""
            llm_resp = self.parse_json_value(span, "gcp.vertex.agent.llm_response")
            if isinstance(llm_resp, dict):
                for part in llm_resp.get("content", {}).get("parts", []):
                    if "text" in part:
                        output_text = part["text"]
                        break
            if not output_text:
                output_data = self.parse_json_value(span, "output.value")
                if isinstance(output_data, dict):
                    for part in output_data.get("content", {}).get("parts", []):
                        if "text" in part:
                            output_text = part["text"]
                            break
            # Fallback for OpenAI instrumentation: output from llm.output_messages
            if not output_text:
                output_text = attrs.get("llm.output_messages.0.message.content", "") or ""

            # Extract tool calls: try llm_response function_call first, then output_messages
            tools_called = []
            if isinstance(llm_resp, dict):
                for part in llm_resp.get("content", {}).get("parts", []):
                    fc = part.get("function_call")
                    if fc:
                        tool_id = fc.get("id", "")
                        tools_called.append(
                            {
                                "tool_name": fc.get("name", ""),
                                "tool_args": fc.get("args", {}),
                                "tool_response": self._find_tool_response(tool_id),
                            }
                        )

            # Fallback: from output_messages attributes (OpenAI instrumentation)
            if not tools_called:
                idx = 0
                while True:
                    tool_name = attrs.get(f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.name")
                    tool_args_str = attrs.get(
                        f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.arguments"
                    )
                    tool_id = attrs.get(f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.id")
                    if tool_name is None:
                        break
                    tools_called.append(
                        {
                            "tool_name": tool_name,
                            "tool_args": json.loads(tool_args_str) if tool_args_str else {},
                            "tool_response": self._find_tool_response(tool_id),
                        }
                    )
                    idx += 1

            # Parse token statistics
            input_tokens = int(attrs.get("gen_ai.usage.input_tokens") or attrs.get("llm.token_count.prompt") or 0)
            output_tokens = int(attrs.get("gen_ai.usage.output_tokens") or attrs.get("llm.token_count.completion") or 0)
            exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time
            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": attrs.get("gen_ai.request.model") or attrs.get("llm.model_name"),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    "start_time": span["start_time_unix_nano"],
                    "end_time": span["end_time_unix_nano"],
                    "span_id": span["span_id"],
                    "trace_id": span["trace_id"],
                    "parent_span_id": span.get("parent_span_id", ""),
                }
            )

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()

        for span in self.trace:
            if not self._is_llm_span(span):
                continue

            attrs = span.get("attributes", {})
            agent_name = self._find_agent_name(span)
            if agent_name == "unknown":
                agent_name = None

            # Extract system prompt: try input_messages first, then llm_request
            if agent_name and agent_name not in agent_settings["prompt"]:
                if attrs.get("llm.input_messages.0.message.role") == "system":
                    system_prompt = attrs.get("llm.input_messages.0.message.content", "")
                    if system_prompt:
                        agent_settings["prompt"][agent_name] = system_prompt
                else:
                    llm_req = self.parse_json_value(span, "gcp.vertex.agent.llm_request")
                    if isinstance(llm_req, dict):
                        sys_instr = llm_req.get("config", {}).get("system_instruction", "")
                        if sys_instr:
                            agent_settings["prompt"][agent_name] = sys_instr

            # Extract tool definitions: try llm.tools first
            idx = 0
            while True:
                tool_schema_str = attrs.get(f"llm.tools.{idx}.tool.json_schema")
                if tool_schema_str is None:
                    break
                try:
                    tool_schema = json.loads(tool_schema_str) if isinstance(tool_schema_str, str) else tool_schema_str
                    # Handle nested {"type":"function","function":{...}} format
                    func_def = tool_schema.get("function", tool_schema)
                    tool_name = func_def.get("name", "")
                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        agent_settings["tool"].append(
                            {
                                "name": tool_name,
                                "description": func_def.get("description", ""),
                                "parameters": func_def.get("parameters", {}),
                            }
                        )
                except Exception:
                    pass
                idx += 1

            # Fallback: tools from llm_request.config.tools[].function_declarations
            if not seen_tools:
                llm_req = self.parse_json_value(span, "gcp.vertex.agent.llm_request")
                if isinstance(llm_req, dict):
                    for tool_group in llm_req.get("config", {}).get("tools", []):
                        for fd in tool_group.get("function_declarations", []):
                            t_name = fd.get("name", "")
                            if t_name and t_name not in seen_tools:
                                seen_tools.add(t_name)
                                agent_settings["tool"].append(
                                    {
                                        "name": t_name,
                                        "description": fd.get("description", ""),
                                        "parameters": fd.get("parameters", {}),
                                    }
                                )

        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # ADK traces: use agent_run spans and descendants
        has_agent_run = any("agent_run" in s.get("name", "") for s in self.trace)

        if has_agent_run:
            agents_with_llm = set()
            for span in self.trace:
                if "call_llm" in span.get("name", ""):
                    name = self._find_agent_name(span)
                    if name != "unknown":
                        agents_with_llm.add(name)

            for agent_span in self.trace:
                if "agent_run" not in agent_span.get("name", ""):
                    continue
                agent_match = re.search(r"agent_run \[(.+?)\]", agent_span.get("name", ""))
                if not agent_match:
                    continue
                agent_name = agent_match.group(1)
                if agent_name not in agents_with_llm:
                    continue
                _ = agent_dependency[agent_name]
                for desc in self._get_adk_descendants(agent_span.get("span_id")):
                    desc_name = desc.get("name", "")
                    if "agent_run" in desc_name:
                        desc_match = re.search(r"agent_run \[(.+?)\]", desc_name)
                        if desc_match:
                            agent_dependency[agent_name]["agent"].append(desc_match.group(1))
                    elif "execute_tool" in desc_name:
                        desc_attrs = desc.get("attributes", {})
                        tool_name = desc_attrs.get("tool.name") or desc_attrs.get("gen_ai.tool.name")
                        if tool_name and tool_name != "(merged tools)":
                            agent_dependency[agent_name]["tool"].append(tool_name)
        else:
            # No agent_run spans (mini_se / OpenAI-only): extract tools from execute_tool spans + LLM output
            for span in self.trace:
                if self._is_llm_span(span):
                    agent_name = self._find_agent_name(span)
                    if agent_name == "unknown":
                        continue
                    _ = agent_dependency[agent_name]
                elif "execute_tool" in span.get("name", ""):
                    # Find which agent this tool belongs to by walking up to a call_llm sibling
                    tool_attrs = span.get("attributes", {})
                    tool_name = tool_attrs.get("tool.name") or tool_attrs.get("gen_ai.tool.name")
                    name_parts = span.get("name", "").replace("execute_tool", "").strip()
                    tool_name = tool_name or name_parts
                    if not tool_name:
                        continue
                    # Attribute tool to the agent found in sibling LLM calls
                    for other in self.trace:
                        if self._is_llm_span(other):
                            agent_name = self._find_agent_name(other)
                            if agent_name != "unknown":
                                agent_dependency[agent_name]["tool"].append(tool_name)
                                break

        # Deduplicate and sort
        return {
            k: {"agent": sorted(set(v["agent"])), "tool": sorted(set(v["tool"]))} for k, v in agent_dependency.items()
        }


import os
import argparse
from pathlib import Path
from collections import defaultdict


def load_spans(case_dir: Path) -> list:
    spans = []
    for f in sorted(case_dir.glob("*.json")):
        if f.name.startswith("._"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for rs in data.get("data", data).get("resource_spans", []):
            for ss in rs.get("scope_spans", []):
                scope_name = ss.get("scope", {}).get("name", "")
                for span in ss.get("spans", []):
                    span["_scope"] = scope_name
                    spans.append(span)
    if not spans:
        return []
    # Group by trace_id, return the largest trace
    by_trace = defaultdict(list)
    for span in spans:
        by_trace[span.get("trace_id", "")].append(span)
    if len(by_trace) > 1:
        print(f"  [WARN] {case_dir.name}: {len(by_trace)} traces found, using largest")
    return max(by_trace.values(), key=len)


def merge_settings(all_settings: list) -> Dict:
    """Merge multiple agent_settings into one, dedup by agent name and tool name."""
    merged = {"prompt": {}, "tool": []}
    seen_tools = set()
    for s in all_settings:
        for agent_name, prompt in s.get("prompt", {}).items():
            if agent_name not in merged["prompt"]:
                merged["prompt"][agent_name] = prompt
        for tool in s.get("tool", []):
            if tool["name"] not in seen_tools:
                seen_tools.add(tool["name"])
                merged["tool"].append(tool)
    return merged


def merge_dependencies(all_deps: list) -> Dict:
    """Merge multiple agent_dependency dicts, dedup agents and tools per agent."""
    merged = defaultdict(lambda: {"agent": set(), "tool": set()})
    for d in all_deps:
        for agent_name, dep in d.items():
            merged[agent_name]["agent"].update(dep.get("agent", []))
            merged[agent_name]["tool"].update(dep.get("tool", []))
    return {k: {"agent": sorted(v["agent"]), "tool": sorted(v["tool"])} for k, v in merged.items()}


def _save(data, path):
    """Write JSON with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {path}")


def parse_trace_dir(trace_dir: Path):
    output_root = trace_dir.parent / "trace_parsed"

    case_dirs = sorted([d for d in trace_dir.iterdir() if d.is_dir() and not d.name.startswith("._")])
    if not case_dirs:
        print(f"No case directories found in {trace_dir}")
        return
    print(f"Found {len(case_dirs)} cases in {trace_dir}")

    all_settings = []
    all_deps = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        spans = load_spans(case_dir)
        if not spans:
            print(f"[{case_name}] No spans found, skipping")
            continue

        case_out = output_root / case_name

        parser = ADKPhoenixParser(spans)
        steps, settings, dependency = parser.parse_all()

        # Save per-trace results
        print(f"[{case_name}] {len(spans)} spans, {len(steps)} steps")
        _save(steps, case_out / "agent_steps.json")
        _save(settings, case_out / "agent_settings.json")
        _save(dependency, case_out / "agent_dependency.json")

        all_settings.append(settings)
        all_deps.append(dependency)

    # Save merged results at trace_parsed/ root
    _save(merge_settings(all_settings), output_root / "agent_settings.json")
    _save(merge_dependencies(all_deps), output_root / "agent_dependency.json")
    print(f"\nMerged settings/dependency saved to {output_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse ADK+autogen Phoenix traces")
    ap.add_argument(
        "--trace_dir",
        type=str,
        default="../results/MBPP/deepseek-v3-2-251201/autogen/trace",
        help="Directory containing Phoenix-exported *.json trace files",
    )
    args = ap.parse_args()
    parse_trace_dir(Path(args.trace_dir))
