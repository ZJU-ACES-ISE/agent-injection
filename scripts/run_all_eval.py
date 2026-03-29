# run_all_eval.py
import argparse
import hashlib
import json
import os
import glob
import logging
import pandas as pd
import re
import sys
import tempfile
import subprocess
import multiprocessing
import signal
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

logger = logging.getLogger("eval_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Default data dir: relative to script location
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "data")

# ── Dataset → evaluation method mapping ──────────────────────────
# "code": execute code + run assert/stdio tests (HumanEval, MBPP, etc.)
# "mcq": execute code or parse text → extract option letter → compare with gt (MMLU-Pro)
# "math": execute code or parse text → math equivalence with gt (MATH)
# "swe": check patch format + optional Docker eval (SWE-bench)
# "general": direct string/numeric comparison with gt (fallback)
EVAL_METHOD = {
    "HumanEval": "code",
    "HumanEval+": "code",
    "MBPP": "code",
    "MBPP+": "code",
    "LiveCodeBench": "code",
    "DS-1000": "code",
    "APPS": "code",
    "MMLU-Pro": "mcq",
    "MATH": "math",
    # "SWE-bench_Pro": "swe",
}
# Shortcut sets derived from EVAL_METHOD (used in evaluate_one and load_dataset)
CODE_DATASETS = {k for k, v in EVAL_METHOD.items() if v == "code"}
SWE_DATASETS = {k for k, v in EVAL_METHOD.items() if v == "swe"}


def extract_code(text: str) -> str:
    # 1) Blocks with any language tag (excludes bare ``` output blocks)
    lang_pattern = r"```(\w+)\s*\n(.*?)```"
    lang_matches = re.findall(lang_pattern, text, re.DOTALL)
    if lang_matches:
        # Last block with real code, excluding test scripts
        for lang, m in reversed(lang_matches):
            if re.search(
                r"^\s*(def |import |from |class |public |private |#include|function |int |void )", m, re.MULTILINE
            ) and not re.search(r"subprocess\.", m):
                return m.strip()
        # Fallback to last tagged block
        return lang_matches[-1][1].strip()
    # 2) Fallback: any fenced code block
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for m in reversed(matches):
            if re.search(
                r"^\s*(def |import |from |class |public |private |#include|function |int |void )", m, re.MULTILINE
            ):
                return m.strip()
        return matches[-1].strip()
    return text.strip()


def _run_code_in_subprocess(code: str, timeout: int = 10, stdin_data: str = None) -> tuple:
    """Run code and return (success, stdout, stderr). Uses Popen with new process group, SIGKILL on timeout."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "main.py")
        with open(p, "w") as f:
            f.write(code)
        try:
            # start_new_session=True: create new process group so we can kill all children (including forked) on timeout
            proc = subprocess.Popen(
                [sys.executable, p],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
                return proc.returncode == 0, stdout, stderr
            except subprocess.TimeoutExpired:
                # SIGKILL the entire process group (including grandchildren spawned via fork)
                logger.debug(f"subprocess timeout ({timeout}s), killing pgid={proc.pid}: {p}")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # proc.pid == process group leader
                except (ProcessLookupError, PermissionError, OSError) as kill_err:
                    logger.debug(f"killpg failed: {kill_err}")
                proc.wait(timeout=5)  # reap zombie process
                return False, "", f"timeout ({timeout}s)"
        except Exception as e:
            logger.debug(f"subprocess error: {e}")
            return False, "", str(e)


def _run_code_get_return(code: str, timeout: int = 10) -> str:
    """Execute code and capture its return value or stdout.
    Strategy: find last defined function, append print(repr(func())) call.
    If code already prints output, use that directly.
    """
    # Find the last 'def funcname(...):' in code
    func_matches = re.findall(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    if func_matches:
        # Append a call to the last function and print its return value
        last_func = func_matches[-1]
        augmented = code + f"\nprint(repr({last_func}()))\n"
        ok, stdout, stderr = _run_code_in_subprocess(augmented, timeout)
        if ok and stdout.strip():
            return stdout.strip().split("\n")[-1]  # last line = repr(return_value)
    # Fallback: run code as-is and capture stdout
    ok, stdout, stderr = _run_code_in_subprocess(code, timeout)
    if ok and stdout.strip():
        return stdout.strip().split("\n")[-1]
    # If execution failed, try to extract return value from code text
    ret_match = re.findall(r'return\s+["\']([A-Ja-j])["\']', code)
    if ret_match:
        return ret_match[-1]  # last return "X" statement
    return ""


def _run_assert_test(code: str, test_case: str, timeout: int = 5) -> bool:
    full = "from typing import *\n" if "from typing import *" not in code else ""
    full += code + "\n" + test_case + "\n"
    ok, _, _ = _run_code_in_subprocess(full, timeout)
    return ok


def _align_func_name(code: str, test_cases: list) -> str:
    """Align model's function name to match test_cases (for MBPP/MBPP+).
    MBPP+ query has no function name, so the model invents its own;
    test_cases assert uses the ground-truth name → NameError. Fix by renaming."""
    if not test_cases:
        return code
    # Extract expected name from first assert, e.g. "assert remove_Occ(...)" or "assert set(func(...))"
    m = re.match(r"assert\s+(?:set|sorted|list|tuple|dict|len|str|int|float|bool|round|max|min|abs)\((\w+)\(", test_cases[0])
    if not m:
        m = re.match(r"assert\s+(\w+)\(", test_cases[0])
    if not m:
        return code
    expected = m.group(1)
    # Extract first def name from model code
    m2 = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    if not m2 or m2.group(1) == expected:
        return code  # already matches or no def found
    # Rename all occurrences of the model's function name to expected
    actual = m2.group(1)
    code = re.sub(r'\b' + re.escape(actual) + r'\b', expected, code)
    logger.debug(f"Aligned func name: {actual} -> {expected}")
    return code


def _run_humaneval_test(code: str, test_str: str, entry_point: str, timeout: int = 10) -> bool:
    full = "from typing import *\n" if "from typing import *" not in code else ""
    full += code + "\n" + test_str + "\n" + f"check({entry_point})\n"
    ok, _, _ = _run_code_in_subprocess(full, timeout)
    return ok


def _run_stdio_test(code: str, input_str: str, expected: str, timeout: int = 10) -> bool:
    ok, stdout, _ = _run_code_in_subprocess(code, timeout, stdin_data=input_str)
    return ok and stdout.strip() == expected.strip()


def get_test_cases(item: dict, dataset: str) -> list:
    if "test_cases" in item and item["test_cases"]:
        return item["test_cases"]
    query = item.get("query", "")
    if dataset in ("HumanEval", "HumanEval+"):
        matches = re.findall(r">>> (.*?)\n\s*([^\n>]*)", query)
        asserts = []
        for call, expected in matches:
            call, expected = call.strip(), expected.strip()
            if expected and not expected.startswith(">>>"):
                asserts.append(f"assert {call} == {expected}")
            elif not expected:
                asserts.append(f"assert {call} is None")
        return asserts
    elif dataset in ("MBPP", "MBPP+"):
        lines = query.split("\n")
        return [lines[2].strip()] if len(lines) > 2 else []
    return []


def eval_code(item: dict, dataset: str) -> tuple:
    """Evaluate code response. Returns (content, score, tests_passed, tests_total)."""
    code = extract_code(item.get("response", ""))
    if dataset == "HumanEval":
        tests, ep = item.get("test_cases", []), item.get("entry_point", "")
        if not tests or not ep:
            return "no test_cases or entry_point", None, 0, 0
        results = [_run_humaneval_test(code, tc, ep) for tc in tests]
        tp, tt = sum(results), len(results)
        return ("passed" if tp == tt else "failed"), (1 if tp == tt else 0), tp, tt
    if dataset == "HumanEval+":
        # test_cases from evalplus are check() functions (same format as HumanEval)
        tests, ep = item.get("test_cases", []), item.get("entry_point", "")
        if not tests or not ep:
            return "no test_cases or entry_point (need evalplus)", None, 0, 0
        results = [_run_humaneval_test(code, tc, ep) for tc in tests]
        tp, tt = sum(results), len(results)
        return ("passed" if tp == tt else "failed"), (1 if tp == tt else 0), tp, tt
    if dataset in ("MBPP", "MBPP+"):
        tests = get_test_cases(item, dataset)
        if not tests:
            return "no test cases", None, 0, 0
        # Align function name: model may use a different name than test_cases expect
        code = _align_func_name(code, tests)
        results = [_run_assert_test(code, tc) for tc in tests]
        tp, tt = sum(results), len(results)
        return ("passed" if tp == tt else "failed"), (1 if tp == tt else 0), tp, tt
    if dataset in ("APPS", "LiveCodeBench"):
        io_raw = item.get("input_output", "")
        if not io_raw:
            return "no input_output", None, 0, 0
        try:
            io = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
        except json.JSONDecodeError as e:
            logger.error(f"invalid input_output JSON: {e}")
            return "invalid input_output json", 0, 0, 0
        pairs = []
        if isinstance(io, dict) and "inputs" in io:
            pairs = [(str(i), str(o)) for i, o in zip(io["inputs"], io["outputs"])]
        elif isinstance(io, list):
            pairs = [(str(tc.get("input", "")), str(tc.get("output", ""))) for tc in io]
        if not pairs:
            return "empty test pairs", None, 0, 0
        results = [_run_stdio_test(code, inp, out) for inp, out in pairs]
        tp, tt = sum(results), len(results)
        return ("passed" if tp == tt else "failed"), (1 if tp == tt else 0), tp, tt
    if dataset == "DS-1000":
        ok, _, err = _run_code_in_subprocess(code, timeout=15)
        return ("passed(syntax)" if ok else f"failed[{err[:100]}]"), (1 if ok else 0), (1 if ok else 0), 1
    return "unsupported dataset", None, 0, 0


def _extract_boxed(s: str) -> str:
    m = re.search(r"\\boxed\{", s)
    if not m:
        return None
    start, depth, i = m.end(), 1, m.end()
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return s[start : i - 1]


def _normalize(s: str) -> str:
    s = s.replace("TERMINATE", "").strip()
    boxed = _extract_boxed(s)
    if boxed is not None:
        s = boxed
    return s.strip().strip("$").strip().lower()


def _extract_final_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    # Match answer up to newline, sentence boundary (but allow decimals like 5.40), or TERMINATE
    m = re.search(r"(?:final answer is|answer is)[:\s]*(.+?)(?:\s*TERMINATE|\n|$)", text, re.I)
    if m:
        # Strip trailing punctuation but preserve decimals: only strip trailing '.' if followed by space or end
        ans = m.group(1).strip().strip("$").strip()
        ans = re.sub(r"\.\s.*", "", ans)  # cut at '. ' (sentence boundary)
        return ans.rstrip(".")
    return text


def eval_general(item: dict) -> tuple:
    gt = _normalize(str(item.get("gt", "")))
    pred = _normalize(_extract_final_answer(str(item.get("response", ""))))
    if not gt:
        return "no ground truth", None
    matched = gt == pred
    if not matched:
        try:
            matched = abs(float(gt) - float(pred)) < 1e-6
        except (ValueError, TypeError):
            pass
    if not matched:
        gt_l = re.search(r"\(([A-Ja-j])\)", gt)
        pred_l = re.search(r"\(([A-Ja-j])\)", pred)
        if gt_l and pred_l:
            matched = gt_l.group(1).upper() == pred_l.group(1).upper()
    return f"gt=[{gt}] pred=[{pred}] -> {'correct' if matched else 'wrong'}", (1 if matched else 0)


def _extract_option_letter(text: str) -> str:
    """Extract multiple-choice option letter (A-J) from text.
    Handles: "(E)", "answer is (E)", "option E", short strings like "E" or "B".
    """
    text = str(text).strip()
    # 1) Parenthesized letter like "(E)" — most reliable pattern
    m = re.search(r"\(([A-Ja-j])\)", text)
    if m:
        return m.group(1).upper()
    # 2) "answer is X" or "correct answer: X"
    m = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-Ja-j])\b", text, re.I)
    if m:
        return m.group(1).upper()
    # 3) Short text (<=5 chars) — likely just the letter itself
    if len(text) <= 5:
        m = re.search(r"([A-Ja-j])", text)
        if m:
            return m.group(1).upper()
    # 4) Last parenthesized letter in longer text (e.g. "... so the answer is (G)")
    matches = re.findall(r"\(([A-Ja-j])\)", text)
    if matches:
        return matches[-1].upper()
    # 5) Dict-like repr: look for 'answer'/'correct_answer' key with single-letter value
    m = re.search(r"""['"](?:correct_)?answer['"]\s*:\s*['"]([A-Ja-j])['"]""", text, re.I)
    if m:
        return m.group(1).upper()
    # 6) Quoted single letter in code output repr, e.g. "'D'" or "('I', 5.4)"
    #    Pick FIRST quoted letter (not last, to avoid dict keys like 'A'..'J')
    quoted = re.findall(r"['\"]([A-Ja-j])['\"]", text)
    if quoted:
        return quoted[0].upper()
    return ""


def _is_code(text: str) -> bool:
    """Check if text looks like Python code (has function defs, imports, etc.)."""
    return bool(re.search(r"^\s*(def |import |from |class )", text, re.MULTILINE))


def _get_answer_from_response(response: str, timeout: int = 15) -> str:
    """Extract answer from response — handles both code and plain text.
    For code: execute and capture return value.
    For plain text: try to extract "answer is X" pattern, else return as-is.
    """
    if not response.strip():
        return ""
    # If response looks like code, execute it to get the answer
    if _is_code(response):
        code = extract_code(response)
        result = _run_code_get_return(code, timeout)
        if result:
            return result
    # Plain text or code execution failed — try to extract final answer
    text = response.strip()
    extracted = _extract_final_answer(text)
    if extracted != text and len(extracted) < len(text):
        return extracted  # successfully extracted a shorter answer
    return text


def eval_mmlu_pro(item: dict) -> tuple:
    """Evaluate MMLU-Pro: extract answer letter from response, compare with gt letter.
    Response can be: code (return "E"), plain letter ("E"), or text ("The answer is (E)...").
    Returns (content, score).
    """
    gt = str(item.get("gt", ""))
    response = str(item.get("response", ""))
    if not gt:
        return "no ground truth", None
    # Extract ground truth letter from "The answer is (I) 5.40MeV"
    gt_letter = _extract_option_letter(gt)
    # Get predicted answer from response (code execution or text)
    pred_raw = _get_answer_from_response(response)
    pred_letter = _extract_option_letter(pred_raw) if pred_raw else ""
    matched = gt_letter == pred_letter and gt_letter != ""
    return (
        f"gt={gt_letter} pred={pred_letter} raw=[{pred_raw[:80]}] -> {'correct' if matched else 'wrong'}",
        1 if matched else 0,
    )


def _latex_to_sympy_str(s: str) -> str:
    """Convert LaTeX math expression to sympy-parseable string."""
    s = str(s).strip()
    # Remove \boxed{...}
    boxed = _extract_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.strip("$").strip()
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # \sqrt{x} -> sqrt(x), \sqrt[n]{x} -> x**(1/n)
    s = re.sub(r"\\sqrt\[([^]]*)\]\{([^}]*)\}", r"((\2))**(1/(\1))", s)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    # Other LaTeX conversions
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/").replace("\\left", "").replace("\\right", "")
    s = s.replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\infty", "oo").replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\cot", "cot").replace("\\tan", "tan").replace("\\sin", "sin")
    s = s.replace("\\cos", "cos").replace("\\ln", "ln").replace("\\log", "log")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)  # \text{abc} -> abc
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # remove remaining unknown LaTeX commands
    s = s.replace("{", "(").replace("}", ")")  # remaining braces to parens
    return s.strip()


def _normalize_math_expr(s: str) -> str:
    """Simple normalize: strip LaTeX and whitespace for direct string comparison."""
    s = _latex_to_sympy_str(s)
    s = re.sub(r"\s+", "", s)  # remove all whitespace
    return s.lower()


def _math_equiv(gt: str, pred: str) -> bool:
    """Check mathematical equivalence between gt (LaTeX) and pred (code output)."""
    # 1) Direct string match after normalization
    gt_n = _normalize_math_expr(gt)
    pred_n = _normalize_math_expr(pred)
    if gt_n == pred_n:
        return True
    # 2) Numeric comparison
    try:
        if abs(float(gt_n) - float(pred_n)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # Also try original strings as floats
    try:
        gt_f = float(gt.strip().strip("$").strip())
        pred_f = float(pred.strip())
        if abs(gt_f - pred_f) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # 3) Sympy symbolic comparison using our LaTeX-to-sympy converter
    try:
        from sympy import sympify, simplify, N

        # Convert gt (LaTeX) and pred to sympy-parseable strings
        gt_sympy_str = _latex_to_sympy_str(gt)
        gt_sym = sympify(gt_sympy_str)
        pred_sym = sympify(pred.strip())
        # Check symbolic equality
        if simplify(gt_sym - pred_sym) == 0:
            return True
        # Check numeric equality
        gt_val, pred_val = complex(N(gt_sym)), complex(N(pred_sym))
        if abs(gt_val - pred_val) < 1e-4:
            return True
    except Exception:
        pass
    return False


def eval_math(item: dict) -> tuple:
    """Evaluate MATH: extract answer from response (code or text), compare with gt.
    Returns (content, score).
    """
    gt = str(item.get("gt", ""))
    response = str(item.get("response", ""))
    if not gt:
        return "no ground truth", None
    # Get predicted answer from response (code execution or text extraction)
    pred_raw = _get_answer_from_response(response)
    if not pred_raw:
        return f"gt=[{gt}] pred=[] (no answer) -> wrong", 0
    # Clean pred: remove repr quotes, try to extract final answer from long text
    pred = pred_raw.strip().strip("'\"")
    # If pred is long text (not a short numeric answer), try to extract the final answer
    if len(pred) > 100:
        pred = _extract_final_answer(pred)
    matched = _math_equiv(gt, pred)
    return (
        f"gt=[{gt}] pred=[{pred}] -> {'correct' if matched else 'wrong'}",
        1 if matched else 0,
    )


def eval_swe_single(item: dict) -> tuple:
    """Pre-check a SWE-bench task before Docker eval.
    Just checks if patch exists and has valid diff format.
    Actual scoring is done by official Docker eval in _try_swebench_harness.
    Returns (content, score, tests_passed, tests_total).
    """
    if item.get("error"):
        return f"error: {item['error']}", 0, 0, 0
    patch = item.get("patch", "").strip()
    if not patch:
        return "no patch generated", 0, 0, 0
    # Valid patch exists — score=0 until Docker eval confirms resolved
    has_diff = "diff --git" in patch or ("---" in patch and "+++" in patch)
    has_hunks = "@@" in patch
    n_files = patch.count("diff --git") or 1
    if has_diff and has_hunks:
        return f"pending docker eval ({n_files} files)", 0, 0, 1
    else:
        return f"invalid patch format", 0, 0, 1


class _EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _EvalTimeout("single task eval timeout")


def evaluate_one(item: dict, dataset: str, timeout: int = 60) -> dict:
    """Dispatch to the correct eval function based on EVAL_METHOD mapping.
    Each single task has a timeout (default 3min). If exceeded, mark as failed.
    """
    result = item.copy()
    method = EVAL_METHOD.get(dataset, "general")
    if "response" not in item or item.get("error"):
        result.update(eval_content="Infer Error", eval_score=None, tests_passed=0, tests_total=0)
        return result
    # Set per-task SIGALRM timeout (only works on Unix main thread; non-main thread raises ValueError)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (ValueError, OSError) as e:
        logger.debug(f"SIGALRM not available (expected in non-main-thread): {e}")
    try:
        if method == "swe":
            result["eval_content"], result["eval_score"], result["tests_passed"], result["tests_total"] = (
                eval_swe_single(item)
            )
        elif method == "code":
            result["eval_content"], result["eval_score"], result["tests_passed"], result["tests_total"] = eval_code(
                item, dataset
            )
        elif method == "mcq":
            result["eval_content"], result["eval_score"] = eval_mmlu_pro(item)
            result["tests_passed"], result["tests_total"] = 0, 0
        elif method == "math":
            result["eval_content"], result["eval_score"] = eval_math(item)
            result["tests_passed"], result["tests_total"] = 0, 0
        else:
            result["eval_content"], result["eval_score"] = eval_general(item)
            result["tests_passed"], result["tests_total"] = 0, 0
    except _EvalTimeout:
        logger.warning(f"Task {item.get('task_id', '?')} eval timeout (>{timeout}s), marking as failed")
        result.update(eval_content="Eval Timeout", eval_score=0, tests_passed=0, tests_total=0)
    finally:
        try:
            signal.alarm(0)  # cancel the alarm
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, OSError) as e:
            logger.debug(f"SIGALRM cleanup failed (non-critical): {e}")
    return result


def discover_experiments(base_dir: str) -> list:
    """Discover experiments from directory structure:
    base_dir/fault_type/dataset/model/method/task_id/output.json
    Also supports: base_dir/dataset/model/method/task_id/output.json (no fault_type layer)
    """
    experiments = []
    seen = set()

    # Try 5-level structure first: fault_type/dataset/model/method/task_id/output.json
    pattern_5 = os.path.join(base_dir, "*", "*", "*", "*", "*", "output.json")
    output_files = sorted(glob.glob(pattern_5))

    if output_files:
        # 5-level: base_dir/fault_type/dataset/model/method/task_id/output.json
        for output_path in output_files:
            parts = output_path.split(os.sep)
            # parts: [..., fault_type, dataset, model, method, task_id, output.json]
            task_id = parts[-2]
            method = parts[-3]
            model = parts[-4]
            dataset = parts[-5]
            fault_type = parts[-6]  # e.g. "results_fault" or "results_nofault"
            # Skip datasets not in EVAL_METHOD (commented out = not evaluated)
            if dataset not in EVAL_METHOD:
                continue

            key = (fault_type, dataset, model, method)
            if key not in seen:
                seen.add(key)
                method_dir = os.path.dirname(os.path.dirname(output_path))
                experiments.append(
                    {
                        "result_dir": fault_type,
                        "fault_type": fault_type,
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "method_dir": method_dir,
                        "structure": "new",
                    }
                )
        logger.info(f"Discovered {len(experiments)} experiments (5-level structure with fault_type)")
        return experiments

    # Fallback: 4-level structure without fault_type
    pattern_4 = os.path.join(base_dir, "*", "*", "*", "*", "output.json")
    output_files = sorted(glob.glob(pattern_4))
    if output_files:
        for output_path in output_files:
            parts = output_path.split(os.sep)
            task_id = parts[-2]
            method = parts[-3]
            model = parts[-4]
            dataset = parts[-5]
            # Skip datasets not in EVAL_METHOD (commented out = not evaluated)
            if dataset not in EVAL_METHOD:
                continue

            key = ("none", dataset, model, method)
            if key not in seen:
                seen.add(key)
                method_dir = os.path.dirname(os.path.dirname(output_path))
                experiments.append(
                    {
                        "result_dir": os.path.basename(base_dir),
                        "fault_type": "none",
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "method_dir": method_dir,
                        "structure": "new",
                    }
                )
        logger.info(f"Discovered {len(experiments)} experiments (4-level structure)")
        return experiments

    logger.warning(f"No experiments found under {base_dir}")
    return experiments


def _load_swe_dataset_hf(max_tasks: int = 300) -> dict:
    """Load SWE-bench Pro from HuggingFace, return {query: item} keyed by problem_statement."""
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("ScaleAI/SWE-bench_Pro", split="test")
        result = {}
        for i, task in enumerate(ds):
            if i >= max_tasks:
                break
            result[task["problem_statement"]] = {
                "query": task["problem_statement"],
                "task_id": task["instance_id"],
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "golden_patch": task.get("patch", ""),  # golden answer patch
                "test_patch": task.get("test_patch", ""),
                "fail_to_pass": task.get("fail_to_pass", ""),
            }
        logger.info(f"Loaded SWE-bench Pro from HF: {len(result)} tasks")
        return result
    except Exception as e:
        logger.warning(f"Cannot load SWE-bench from HF: {e}")
        return {}


def load_dataset(data_dir: str, dataset: str) -> dict:
    """Load dataset ground truth, return {query: item}."""
    if dataset in SWE_DATASETS:
        return _load_swe_dataset_hf()
    ds_path = os.path.join(data_dir, f"{dataset}.json")
    if not os.path.exists(ds_path):
        logger.error(f"Dataset file not found: {ds_path}")
        return {}
    with open(ds_path) as f:
        items = {item["query"]: item for item in json.load(f)}
    # Enrich HumanEval+ with evalplus test suite (dataset json has no test_cases)
    if dataset == "HumanEval+":
        try:
            from evalplus.data import get_human_eval_plus

            evalplus_data = get_human_eval_plus()
            # Build mapping: entry_point or task_id -> evalplus item
            ep_by_task = {v["task_id"]: v for v in evalplus_data.values()}
            enriched = 0
            for query, item in items.items():
                # Match by task_id (e.g. "HumanEval/155")
                tid = item.get("task_id", "")
                ep_item = ep_by_task.get(tid)
                if ep_item and "test" in ep_item:
                    # Use evalplus's check() function as test_cases
                    item["test_cases"] = [ep_item["test"]]
                    enriched += 1
            logger.info(f"Enriched HumanEval+ with evalplus test suite: {enriched}/{len(items)} tasks")
        except Exception as e:
            logger.warning(f"Failed to load evalplus for HumanEval+: {e}")
    return items


def load_new_structure_results(method_dir: str) -> dict:
    """Load results from new structure: method_dir/task_id/output.json"""
    results = {}
    if not os.path.isdir(method_dir):
        return results

    for task_id in os.listdir(method_dir):
        task_dir = os.path.join(method_dir, task_id)
        if not os.path.isdir(task_dir):
            continue

        output_file = os.path.join(task_dir, "output.json")
        input_raw_file = os.path.join(task_dir, "input_raw.json")

        if not os.path.isfile(output_file):
            continue

        try:
            with open(output_file) as f:
                output_data = json.load(f)

            # Load original query from input_raw.json
            query = ""
            if os.path.isfile(input_raw_file):
                with open(input_raw_file) as f:
                    input_data = json.load(f)
                query = input_data.get("query") or input_data.get("prompt") or ""

            # Build result item
            item = {
                "task_id": task_id,
                "query": query,
                "response": output_data.get("final_answer", ""),
                "patch": output_data.get("patch", ""),  # SWE-bench patch
                "error": output_data.get("error"),
                "_fault_name": output_data.get("_fault_name"),
                "_fault_fired": output_data.get("_fault_fired"),
            }

            # Use query as key (for matching with dataset ground truth)
            if query:
                results[query] = item
            else:
                results[task_id] = item

        except Exception as e:
            logger.error(f"Failed to load {output_file}: {e}")

    return results


def eval_one_experiment(exp: dict, data_dir: str) -> list:
    """Evaluate one experiment, return list of row dicts."""
    dataset = exp["dataset"]
    all_queries = load_dataset(data_dir, dataset)
    if not all_queries:
        logger.warning(f"Empty dataset {dataset}, skip {exp}")
        return []

    # Load inferred results based on structure type
    inferred = {}
    if exp.get("structure") == "new":
        inferred = load_new_structure_results(exp["method_dir"])
    else:
        # Old structure: infer.jsonl
        try:
            with open(exp["infer_path"]) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        inferred[item["query"]] = item
        except Exception as e:
            logger.error(f"Failed to load {exp['infer_path']}: {e}")
            return []

    # For SWE-bench, match by task_id (query is too long/unstable for matching)
    if dataset in SWE_DATASETS:
        return _eval_swe_experiment(exp, inferred, all_queries)

    logger.info(f"[{exp['method']}|{dataset}|{exp['model']}] " f"inferred={len(inferred)}/{len(all_queries)}")

    rows = []
    exp_label = f"{exp['method']}|{dataset}|{exp['model']}"  # log prefix
    total = len(all_queries)
    correct_count = 0
    for i, (query, original) in enumerate(all_queries.items(), 1):
        if query in inferred:
            # Merge dataset ground truth into inferred result (gt, test_cases, entry_point, etc.)
            merged = {**original, **inferred[query]}
            r = evaluate_one(merged, dataset)
        else:
            r = original.copy()
            r.update({"eval_content": "Not Inferred", "eval_score": 0, "tests_passed": 0, "tests_total": 0})

        score = r.get("eval_score")
        if score == 1:
            correct_count += 1
        # Log progress every 50 tasks or at the last task (with cumulative accuracy)
        if i % 50 == 0 or i == total:
            logger.info(f"[{exp_label}] progress {i}/{total}, correct so far: {correct_count}/{i}")

        rows.append(
            {
                "fault_type": exp.get("fault_type", "none"),
                "result_dir": exp["result_dir"],
                "model": exp["model"],
                "method": exp["method"],
                "dataset": dataset,
                "task_id": r.get("task_id", ""),
                "query": r.get("query", "")[:200],
                "gt": str(r.get("gt", ""))[:200],
                "response": str(r.get("response", ""))[:500],
                "error": str(r.get("error", ""))[:300],
                "eval_content": r.get("eval_content", ""),
                "eval_score": r.get("eval_score"),
                "tests_passed": r.get("tests_passed", ""),
                "tests_total": r.get("tests_total", ""),
                "_fault_name": r.get("_fault_name", ""),
                "_fault_fired": r.get("_fault_fired", ""),
            }
        )
    return rows


def _eval_swe_experiment(exp: dict, inferred: dict, all_queries: dict) -> list:
    """Evaluate SWE-bench experiment: match by task_id, collect predictions, try harness."""
    dataset = exp["dataset"]
    # Build task_id index for both sides
    gt_by_tid = {v.get("task_id") or v.get("instance_id", ""): v for v in all_queries.values()}
    inf_by_tid = {}
    for v in inferred.values():
        tid = v.get("task_id", "")
        if tid:
            inf_by_tid[tid] = v

    logger.info(
        f"[{exp['method']}|{dataset}|{exp['model']}] "
        f"inferred={len(inf_by_tid)}/{len(gt_by_tid)} (matched by task_id)"
    )

    # Collect predictions.jsonl for swebench harness
    pred_lines = []
    rows = []
    for tid, gt_item in gt_by_tid.items():
        inf_item = inf_by_tid.get(tid)
        if inf_item:
            merged = {**gt_item, **inf_item}
            r = evaluate_one(merged, dataset)
            # Collect for harness eval
            patch = inf_item.get("patch", "").strip()
            pred_lines.append(
                {
                    "instance_id": gt_item.get("instance_id", tid),
                    "model_name_or_path": f"{exp['method']}-{exp['model']}",
                    "model_patch": patch,
                }
            )
        else:
            r = gt_item.copy()
            r.update({"eval_content": "Not Inferred", "eval_score": 0, "tests_passed": 0, "tests_total": 0})

        rows.append(
            {
                "fault_type": exp.get("fault_type", "none"),
                "result_dir": exp["result_dir"],
                "model": exp["model"],
                "method": exp["method"],
                "dataset": dataset,
                "task_id": r.get("task_id", tid),
                "query": r.get("query", "")[:200],
                "gt": str(r.get("instance_id", ""))[:200],  # SWE: gt = instance_id
                "response": str(r.get("patch", r.get("response", "")))[:500],
                "error": str(r.get("error", ""))[:300],
                "eval_content": r.get("eval_content", ""),
                "eval_score": r.get("eval_score"),
                "tests_passed": r.get("tests_passed", ""),
                "tests_total": r.get("tests_total", ""),
                "_fault_name": r.get("_fault_name", ""),
                "_fault_fired": r.get("_fault_fired", ""),
            }
        )

    # Try swebench harness evaluation
    if pred_lines:
        _try_swebench_harness(exp, pred_lines, rows)

    return rows


# ── SWE-bench Pro Docker evaluation (self-contained) ─────────────

_swe_meta_cache: dict = {}  # {instance_id: row_dict} from JSONL


_swe_repo_root: str = ""  # root of SWE-bench_Pro-os repo


def _load_swe_meta() -> dict:
    """Load sweap_eval_full_v2.jsonl as {instance_id: metadata}. Cached."""
    global _swe_repo_root
    if _swe_meta_cache:
        return _swe_meta_cache
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, "..", "SWE-bench_Pro-os", "helper_code", "sweap_eval_full_v2.jsonl"),
        os.path.join(script_dir, "sweap_eval_full_v2.jsonl"),
    ]:
        if os.path.isfile(candidate):
            _swe_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(candidate)))
            with open(candidate) as f:
                for line in f:
                    row = json.loads(line)
                    _swe_meta_cache[row["instance_id"]] = row
            logger.info(
                f"Loaded SWE meta: {len(_swe_meta_cache)} instances from {candidate} (repo_root={_swe_repo_root})"
            )
            return _swe_meta_cache
    logger.warning("sweap_eval_full_v2.jsonl not found — Docker eval unavailable")
    return _swe_meta_cache


def _swe_dockerhub_uri(instance_id: str, repo: str, username: str = "jefzda") -> str:
    """Generate Docker Hub image URI from instance_id + repo. Mirrors official image_uri.py."""
    repo_base, repo_name_only = repo.lower().split("/")
    hsh = instance_id.replace("instance_", "")
    # Special cases matching official logic
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo.lower() and "element-web" in repo.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"[:128]
    return f"{username}/sweap-images:{tag}"


def _swe_build_entryscript(meta: dict, patch_path: str) -> str:
    """Build the bash entry script that runs inside the Docker container."""
    base_commit = meta["base_commit"]
    instance_id = meta["instance_id"]
    before_cmd = meta.get("before_repo_set_cmd", "").strip().split("\n")[-1]
    test_files = ",".join(json.loads(meta["selected_test_files_to_run"]))
    # Extract ENV from local Dockerfile files (JSONL meta stores S3 URLs, not actual content)
    env_lines = []
    for df_subdir in ["base_dockerfile", "instance_dockerfile"]:
        df_path = os.path.join(_swe_repo_root, "dockerfiles", df_subdir, instance_id, "Dockerfile")
        if os.path.isfile(df_path):
            with open(df_path) as df_f:
                for line in df_f:
                    line = line.strip()
                    if line.startswith("ENV"):
                        env_lines.append(line.replace("ENV", "export", 1))
        else:
            logger.warning(f"{instance_id}: Dockerfile not found: {df_path}")
    logger.info(
        f"[entryscript] {instance_id}: commit={base_commit[:12]} test_files={test_files} env_count={len(env_lines)} env={env_lines} before_cmd={before_cmd!r}"
    )
    env_block = "\n".join(env_lines)
    script = f"""
{env_block}
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
echo "=== git apply ===" > /workspace/apply.log 2>&1
git apply -v /workspace/patch.diff >> /workspace/apply.log 2>&1
echo "=== git apply exit code: $? ===" >> /workspace/apply.log
{before_cmd}
bash /workspace/run_script.sh {test_files} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    logger.info(f"[entryscript] {instance_id}: full script:\n{script}")
    return script


def _swe_strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept = []
    for s in sections:
        if not s.strip():
            continue
        if re.search(r"^Binary files .* differ$", s, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", s, re.MULTILINE):
            continue
        kept.append(s)
    return "".join(kept)


def _swe_eval_one_docker(instance_id: str, patch: str, meta: dict) -> dict | None:
    """Run one SWE-bench instance in Docker. Returns {"resolved": bool, "passed": int, "f2p": int, "p2p": int} or None."""
    try:
        import docker as docker_sdk
    except ImportError:
        logger.error("docker SDK not installed — pip install docker")
        return None

    repo = meta.get("repo", meta.get("repo_name", ""))
    if not repo or "/" not in repo:
        logger.error(f"{instance_id}: no valid repo field in meta")
        return None

    image_uri = _swe_dockerhub_uri(instance_id, repo)
    cleaned_patch = _swe_strip_binary_hunks(patch)
    logger.info(f"[docker] {instance_id}: repo={repo} image={image_uri} patch_len={len(cleaned_patch)} patch_first_line={cleaned_patch.split(chr(10))[0]!r}")

    with tempfile.TemporaryDirectory(prefix="swe_docker_") as workspace:
        # Write workspace files (ensure trailing newline — git apply rejects patches without it)
        with open(os.path.join(workspace, "patch.diff"), "w") as f:
            f.write(cleaned_patch if cleaned_patch.endswith("\n") else cleaned_patch + "\n")
        # Load run_script.sh and parser.py from local run_scripts/ dir
        scripts_dir = os.path.join(_swe_repo_root, "run_scripts", instance_id)
        for fname in ["run_script.sh", "parser.py"]:
            src = os.path.join(scripts_dir, fname)
            if os.path.isfile(src):
                with open(src) as sf:
                    content = sf.read()
            else:
                logger.error(f"[docker] {instance_id}: {src} not found")
                return None
            with open(os.path.join(workspace, fname), "w") as f:
                f.write(content)
        entry = _swe_build_entryscript(meta, os.path.join(workspace, "patch.diff"))
        with open(os.path.join(workspace, "entryscript.sh"), "w") as f:
            f.write(entry)

        # Run Docker container
        client = docker_sdk.from_env()
        try:
            import platform as py_platform
            plat = "linux/amd64" if py_platform.machine().lower() in {"arm64", "aarch64"} else None
            logger.info(f"[docker] {instance_id}: pulling {image_uri} platform={plat}")
            try:
                client.images.pull(image_uri, platform=plat) if plat else client.images.pull(image_uri)
            except Exception as pull_err:
                try:
                    client.images.get(image_uri)
                    logger.info(f"[docker] {instance_id}: using local image (pull failed: {pull_err})")
                except Exception:
                    logger.error(f"[docker] {instance_id}: failed to pull {image_uri}: {pull_err}")
                    return None

            abs_ws = os.path.abspath(workspace)
            run_kwargs = {
                "volumes": {abs_ws: {"bind": "/workspace", "mode": "rw"}},
                "detach": True,
                "remove": True,
                "entrypoint": "/bin/bash",
                "command": ["-c", "bash /workspace/entryscript.sh"],
            }
            if plat:
                run_kwargs["platform"] = plat

            logger.info(f"[docker] {instance_id}: starting container, workspace={abs_ws}")
            container = client.containers.run(image_uri, **run_kwargs)
            result = container.wait(timeout=900)
            rc = result.get("StatusCode", 1) if isinstance(result, dict) else 1
            logger.info(f"[docker] {instance_id}: container finished rc={rc}")
            # Always log apply/stdout/stderr tail for debugging
            for logf in ["apply.log", "stdout.log", "stderr.log"]:
                logpath = os.path.join(workspace, logf)
                if os.path.isfile(logpath):
                    with open(logpath) as lf:
                        content = lf.read().strip()
                    if content:
                        logger.info(f"[docker] {instance_id} {logf} (last 300): {content[-300:]}")
        except Exception as e:
            logger.error(f"[docker] {instance_id}: Docker error: {e}")
            return None

        # Parse output.json
        output_file = os.path.join(workspace, "output.json")
        if not os.path.isfile(output_file):
            logger.error(f"[docker] {instance_id}: no output.json produced, workspace files: {os.listdir(workspace)}")
            return None
        try:
            with open(output_file) as f:
                output = json.load(f)
        except Exception as e:
            logger.error(f"[docker] {instance_id}: failed to parse output.json: {e}")
            return None

        # Evaluate: (fail_to_pass ∪ pass_to_pass) ⊆ passed_tests
        all_tests = output.get("tests", [])
        passed_tests = {t["name"] for t in all_tests if t.get("status") == "PASSED"}
        failed_tests = {t["name"] for t in all_tests if t.get("status") == "FAILED"}
        logger.info(f"[docker] {instance_id}: output.json has {len(all_tests)} tests, {len(passed_tests)} PASSED, {len(failed_tests)} FAILED")
        try:
            raw_f2p = meta.get("FAIL_TO_PASS", "[]")
            raw_p2p = meta.get("PASS_TO_PASS", "[]")
            f2p = set(json.loads(raw_f2p) if isinstance(raw_f2p, str) else raw_f2p)
            p2p = set(json.loads(raw_p2p) if isinstance(raw_p2p, str) else raw_p2p)
        except Exception as e:
            logger.error(f"[docker] {instance_id}: bad FAIL_TO_PASS/PASS_TO_PASS: {e}")
            return None
        # Log exactly which required tests are missing from passed
        f2p_missing = f2p - passed_tests
        p2p_missing = p2p - passed_tests
        if f2p_missing:
            logger.warning(f"[docker] {instance_id}: f2p NOT passed ({len(f2p_missing)}/{len(f2p)}): {f2p_missing}")
        else:
            logger.info(f"[docker] {instance_id}: f2p ALL passed ({len(f2p)}): {f2p}")
        if p2p_missing:
            logger.warning(f"[docker] {instance_id}: p2p NOT passed ({len(p2p_missing)}/{len(p2p)}): {list(p2p_missing)[:5]}...")
        resolved = (f2p | p2p) <= passed_tests
        logger.info(f"[docker] {instance_id}: {'RESOLVED' if resolved else 'FAILED'} passed={len(passed_tests)} f2p={len(f2p)}(missing={len(f2p_missing)}) p2p={len(p2p)}(missing={len(p2p_missing)})")
        return {"resolved": resolved, "passed": len(passed_tests), "f2p": len(f2p), "p2p": len(p2p)}


_swe_cache_base: str = ""  # set by main() from --output arg


def _swe_cache_dir() -> str:
    """Return (and create) the SWE-bench Docker eval cache directory."""
    base = _swe_cache_base or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_paper")
    d = os.path.join(base, ".swe_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _swe_cache_key(instance_id: str, patch: str, exp: dict) -> str:
    """Generate cache filename from instance_id + patch_hash + experiment identity (fault_type/model/method)."""
    # Include experiment identity so fault vs nofault / different models never collide
    exp_tag = f"{exp.get('fault_type', 'none')}_{exp.get('model', '')}_{exp.get('method', '')}"
    content_hash = hashlib.md5((patch + "|" + exp_tag).encode()).hexdigest()[:12]
    return f"{instance_id}_{content_hash}.json"


def _swe_cache_lookup(instance_id: str, patch: str, exp: dict) -> dict | None:
    """Check cache. Returns {"resolved": bool, "passed": int, "f2p": int, "p2p": int} or None."""
    cache_file = os.path.join(_swe_cache_dir(), _swe_cache_key(instance_id, patch, exp))
    if os.path.isfile(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"SWE cache read failed for {instance_id} ({cache_file}): {e}")
    return None


def _swe_cache_save(
    instance_id: str, patch: str, exp: dict, resolved: bool, passed: int = 0, f2p: int = 0, p2p: int = 0
):
    """Save a Docker eval result to cache with full experiment identity."""
    cache_file = os.path.join(_swe_cache_dir(), _swe_cache_key(instance_id, patch, exp))
    try:
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "instance_id": instance_id,
                    "fault_type": exp.get("fault_type", "none"),
                    "model": exp.get("model", ""),
                    "method": exp.get("method", ""),
                    "resolved": resolved,
                    "passed": passed,
                    "f2p": f2p,
                    "p2p": p2p,
                },
                f,
            )
    except Exception as e:
        logger.warning(f"Failed to save SWE cache for {instance_id}: {e}")


def _try_swebench_harness(exp: dict, pred_lines: list, rows: list):
    """Run SWE-bench Pro Docker eval for all non-empty patches. Updates rows incrementally.
    Results are cached per (instance_id, patch_hash) so interrupted runs can resume.
    """
    non_empty = [p for p in pred_lines if p["model_patch"].strip()]
    exp_label = f"{exp['method']}|{exp['dataset']}|{exp['model']}"
    logger.info(f"[{exp_label}] SWE patches: {len(non_empty)}/{len(pred_lines)} non-empty")
    if not non_empty:
        return

    meta_index = _load_swe_meta()
    if not meta_index:
        logger.warning("No SWE meta loaded, skipping Docker eval")
        return

    # Check Docker availability
    docker_ok = True
    try:
        import docker as docker_sdk

        docker_sdk.from_env().ping()
    except Exception as e:
        logger.warning(f"Docker not available ({e}), will only use cached results")
        docker_ok = False

    # Build row index for fast lookup: {instance_id: [row_indices]}
    row_by_iid = {}
    for i, row in enumerate(rows):
        tid = row.get("task_id", "")
        gt_id = row.get("gt", "")
        for key in (tid, gt_id):
            if key:
                row_by_iid.setdefault(key, []).append(i)

    # Evaluate each patch incrementally (with cache)
    resolved_count = 0
    cached_count = 0
    for idx, p in enumerate(non_empty, 1):
        iid = p["instance_id"]
        patch = p["model_patch"]

        # Check cache first (keyed by instance_id + patch + experiment identity)
        cached = _swe_cache_lookup(iid, patch, exp)
        if cached is not None:
            is_resolved = cached.get("resolved", False)
            detail = f"passed={cached.get('passed',0)} f2p={cached.get('f2p',0)} p2p={cached.get('p2p',0)}"
            cached_count += 1
            tag = "cached"
        elif not docker_ok:
            is_resolved = False
            detail = "no-docker"
            tag = "no-docker"
        else:
            meta = meta_index.get(iid)
            if not meta:
                logger.warning(f"[{exp_label}] [{idx}/{len(non_empty)}] {iid}: not in meta, skip")
                is_resolved = False
                detail = "no-meta"
                tag = "no-meta"
            else:
                result = _swe_eval_one_docker(iid, patch, meta)
                if result is not None:
                    is_resolved = result["resolved"]
                    detail = f"passed={result['passed']} f2p={result['f2p']} p2p={result['p2p']}"
                    _swe_cache_save(iid, patch, exp, is_resolved, result["passed"], result["f2p"], result["p2p"])
                else:
                    is_resolved = False
                    detail = "docker-error"
                tag = "docker"

        if is_resolved:
            resolved_count += 1

        # Update corresponding rows immediately
        status = "RESOLVED" if is_resolved else "FAILED"
        for row_idx in row_by_iid.get(iid, []):
            rows[row_idx]["eval_content"] = f"{status} ({detail})"
            rows[row_idx]["eval_score"] = 1 if is_resolved else 0
            rows[row_idx]["tests_passed"] = 1 if is_resolved else 0
            rows[row_idx]["tests_total"] = 1

        logger.info(
            f"[{exp_label}] [{idx}/{len(non_empty)}] {iid}: {'RESOLVED' if is_resolved else 'FAILED'} [{tag}] ({detail}) (cumulative {resolved_count}/{idx})"
        )

    logger.info(
        f"[{exp_label}] SWE Docker eval done: {resolved_count}/{len(non_empty)} resolved ({cached_count} from cache)"
    )


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary from detail DataFrame."""
    df["eval_score"] = pd.to_numeric(df["eval_score"], errors="coerce")
    group_cols = ["fault_type", "dataset", "model", "method"]
    g = df.groupby(group_cols)
    summary = g.agg(
        total=("eval_score", "size"),
        valid=("eval_score", "count"),
        correct=("eval_score", "sum"),
    ).reset_index()
    summary["accuracy"] = (summary["correct"] / summary["total"] * 100).round(2).astype(str) + "%"

    # Count error categories per group
    err_counts = (
        df.groupby(group_cols)
        .apply(
            lambda g: pd.Series(
                {
                    "infer_errors": (g["eval_content"] == "Infer Error").sum(),
                    "not_inferred": (g["eval_content"] == "Not Inferred").sum(),
                }
            )
        )
        .reset_index()
    )
    summary = summary.merge(err_counts, on=group_cols)
    return summary


def get_experiment_key(exp: dict) -> tuple:
    """Get unique key for an experiment."""
    return (exp.get("fault_type", "none"), exp["dataset"], exp["model"], exp["method"])


def load_existing_results(detail_path: str) -> tuple:
    """Load existing eval_detail.csv, return (DataFrame, set of completed experiment keys)."""
    if not os.path.exists(detail_path):
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(detail_path)
        if df.empty:
            return pd.DataFrame(), set()
        # Group by experiment keys to find completed ones
        completed = set()
        # Ensure fault_type column exists for backward compat
        if "fault_type" not in df.columns:
            df["fault_type"] = "none"
        for _, group in df.groupby(["fault_type", "dataset", "model", "method"]):
            key = (
                group["fault_type"].iloc[0],
                group["dataset"].iloc[0],
                group["model"].iloc[0],
                group["method"].iloc[0],
            )
            completed.add(key)
        logger.info(f"Loaded {len(df)} existing results, {len(completed)} experiments completed")
        return df, completed
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return pd.DataFrame(), set()


# Global lock for CSV writing
_csv_write_lock = threading.Lock()


def append_rows_to_csv(detail_path: str, rows: list, write_header: bool = False):
    """Append rows to CSV file immediately (thread-safe)."""
    if not rows:
        return
    with _csv_write_lock:
        df = pd.DataFrame(rows)
        df.to_csv(detail_path, mode="a", header=write_header, index=False)


def _eval_experiment_wrapper(exp: dict, data_dir: str) -> tuple:
    """Wrapper for parallel execution. Returns (exp_info, rows, error)."""
    exp_info = f"{exp.get('fault_type','')}/{exp['dataset']}/{exp['model']}/{exp['method']}"
    try:
        rows = eval_one_experiment(exp, data_dir)
        return exp_info, rows, None
    except Exception as e:
        return exp_info, [], str(e)


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation across all result directories")
    parser.add_argument(
        "--base_dir", default="../results_all", help="Parent dir containing results_fault/results_nofault folders"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dataset dir (default: ../datasets/data)")
    parser.add_argument("--output", default="../results_paper", help="Output dir for csv files")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: min(cpu_count, 8, num_experiments))",
    )
    parser.add_argument("--dataset", default=None, help="Evaluate only this dataset (e.g. SWE-bench_Pro)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation, ignore existing results")
    args = parser.parse_args()

    # Resolve all paths to absolute (important for subprocess workers)
    args.base_dir = os.path.abspath(args.base_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    args.output = os.path.abspath(args.output)

    # Set SWE cache dir based on output dir
    global _swe_cache_base
    _swe_cache_base = args.output

    # Discover all experiments
    experiments = discover_experiments(args.base_dir)
    if args.dataset:
        experiments = [e for e in experiments if e["dataset"] == args.dataset]
        logger.info(f"Filtered to dataset={args.dataset}: {len(experiments)} experiments")
    if not experiments:
        logger.warning("No experiments found, exiting.")
        return

    # Load existing results
    os.makedirs(args.output, exist_ok=True)
    detail_path = os.path.join(args.output, "eval_detail.csv")

    if args.force and os.path.exists(detail_path):
        os.remove(detail_path)  # clean slate
        logger.info(f"Force mode: removed existing {detail_path}")
    existing_df, completed_keys = (pd.DataFrame(), set()) if args.force else load_existing_results(detail_path)

    # Filter out already completed experiments
    pending_experiments = []
    for exp in experiments:
        key = get_experiment_key(exp)
        if key in completed_keys:
            logger.info(
                f"[SKIP] {exp.get('fault_type','')}/{exp['dataset']}/{exp['model']}/{exp['method']} (already evaluated)"
            )
        else:
            pending_experiments.append(exp)

    if not pending_experiments:
        logger.info("All experiments already evaluated, nothing to do.")
        # Still rebuild summary from existing data
        if not existing_df.empty:
            summary = build_summary(existing_df)
            summary_path = os.path.join(args.output, "eval_summary.csv")
            summary.to_csv(summary_path, index=False)
            print(f"\n{'='*100}")
            print(summary.to_string(index=False))
            print(f"{'='*100}")
        return

    # Determine number of workers
    max_workers = args.workers or min(multiprocessing.cpu_count(), 8, len(pending_experiments))
    logger.info(
        f"Evaluating {len(pending_experiments)}/{len(experiments)} experiments with {max_workers} parallel workers"
    )

    # Check if we need to write header (new file or empty existing)
    need_header = existing_df.empty

    # Evaluate pending experiments in parallel, save incrementally
    new_rows_count = 0
    eval_func = partial(_eval_experiment_wrapper, data_dir=args.data_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {}
        for i, exp in enumerate(pending_experiments, 1):
            fut = executor.submit(eval_func, exp)
            future_to_exp[fut] = (i, exp)

        for future in as_completed(future_to_exp):
            idx, exp = future_to_exp[future]
            exp_label = f"{exp.get('fault_type','')}/{exp['dataset']}/{exp['model']}/{exp['method']}"
            try:
                exp_info, rows, error = future.result()
                if error:
                    logger.error(f"[{idx}/{len(pending_experiments)}] {exp_info} failed: {error}")
                else:
                    # Save immediately after each experiment completes
                    append_rows_to_csv(detail_path, rows, write_header=need_header)
                    need_header = False  # Only write header once
                    new_rows_count += len(rows)
                    logger.info(f"[{idx}/{len(pending_experiments)}] {exp_info} done, {len(rows)} samples (saved)")
            except Exception as e:
                logger.error(f"[{idx}/{len(pending_experiments)}] {exp_label} unexpected error: {e}")

    # Reload full results and build summary
    df = pd.read_csv(detail_path) if os.path.exists(detail_path) else pd.DataFrame()
    if df.empty:
        logger.warning("No results to summarize")
        return
    summary = build_summary(df)

    # Save summary
    summary_path = os.path.join(args.output, "eval_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Print summary
    print(f"\n{'='*100}")
    print(summary.to_string(index=False))
    print(f"{'='*100}")
    print(f"\nDetail  -> {detail_path}")
    print(f"Summary -> {summary_path}")
    print(
        f"Total: {len(experiments)} experiments, {new_rows_count} new + {len(existing_df)} existing = {len(df)} samples"
    )


if __name__ == "__main__":
    main()
