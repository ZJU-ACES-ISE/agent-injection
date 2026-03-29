# prepare_dataset.py
import json
import os
import random
import argparse

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/data_cached")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = _cache_dir

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MATH")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--num2sample", type=int, default=300)
args = parser.parse_args()
save_path = os.path.join("../datasets/data", f"{args.dataset_name}.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)


if os.path.exists(save_path):
    print(f">> {save_path} already exists, skip. Delete it to re-generate.")
    exit(0)


def shuffle_and_sample(data_list, num2sample):
    random.seed(42)
    random.shuffle(data_list)
    return data_list[:num2sample]


def deduplicate(data_list):
    seen_queries = set()
    unique_data = []

    for item in data_list:
        if item["query"] not in seen_queries:
            unique_data.append(item)  # keep the first occurrence
            seen_queries.add(item["query"])  # mark this query as seen
    if len(unique_data) < len(data_list):
        print(f">> Duplicate samples removed: {len(data_list) - len(unique_data)}")
    return unique_data


# load MATH-500 dataset
if args.dataset_name == "MATH":
    load_dataset_path = args.dataset_path if args.dataset_path else "HuggingFaceH4/MATH-500"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [
                args.dataset_name,
                "math",
                example["subject"],
                f"Level {example['level']}",
            ],
            "source": args.dataset_name,
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    for i, item in enumerate(data_list):
        item["task_id"] = f"MATH_{i:04d}"

# load GSM8K dataset
elif args.dataset_name == "GSM8K":
    load_dataset_path = args.dataset_path if args.dataset_path else "openai/gsm8k"
    dataset = load_dataset(load_dataset_path, "main", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {"query": example["question"], "gt": example["answer"], "tag": ["math"], "source": "GSM8K"}
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load AQUA-RAT dataset
elif args.dataset_name == "AQUA-RAT":
    load_dataset_path = args.dataset_path if args.dataset_path else "deepmind/aqua_rat"
    dataset = load_dataset(load_dataset_path, "raw", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)  # question / options / rationale / correct

    def format_aqua_rat_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n{option}"
        return query

    data_list = [
        {
            "query": format_aqua_rat_query(example),
            "gt": example["rationale"],
            "tag": ["math", "reasoning", "multiple-choice"],
            "source": "AQUA-RAT",
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedMCQA
elif args.dataset_name == "MedMCQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "openlifescienceai/medmcqa"
    dataset = load_dataset(load_dataset_path, split="validation", trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda example: example["choice_type"] != "multi")
    print(f"{'='*50}\n", filtered_dataset)

    def format_medmcqa_query(example):
        query = example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['opa']}"
        query += f"\n(B) {example['opb']}"
        query += f"\n(C) {example['opc']}"
        query += f"\n(D) {example['opd']}"
        return query

    def format_medmcqa_gt(example):
        answer_list = [
            f"(A) {example['opa']}",
            f"(B) {example['opb']}",
            f"(C) {example['opc']}",
            f"(D) {example['opd']}",
        ]
        answer = f"The correct answer is: {answer_list[example['cop']]}"
        return answer

    data_list = [
        {
            "query": format_medmcqa_query(example),
            "gt": format_medmcqa_gt(example),
            "tag": ["medical", example["subject_name"], example["topic_name"]],
            "source": "MedMCQA",
        }
        for example in filtered_dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedQA
elif args.dataset_name == "MedQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "GBaker/MedQA-USMLE-4-options"
    dataset = load_dataset(load_dataset_path, split="test")
    print(f"{'='*50}\n", dataset)

    def format_medqa_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for key, value in example["options"].items():
            query += f"\n({key}) {value}"
        return query

    def format_medqa_gt(example):
        ans_key = [k for k, v in example["options"].items() if v == example["answer"]][0]
        answer = f"The correct answer is: ({ans_key}) {example['answer']}"
        return answer

    data_list = [
        {"query": format_medqa_query(example), "gt": format_medqa_gt(example), "tag": ["medical"], "source": "MedQA"}
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU
elif args.dataset_name == "MMLU":
    load_dataset_path = args.dataset_path if args.dataset_path else "cais/mmlu"
    dataset = load_dataset(load_dataset_path, "all", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_mmlu_query(example):
        query = f"""The following is a multiple-choice question:
{example["question"]}

Choose the correct answer from the following options:
(A) {example["choices"][0]}
(B) {example["choices"][1]}
(C) {example["choices"][2]}
(D) {example["choices"][3]}"""
        return query

    def format_mmlu_gt(example):
        choice_list = ["A", "B", "C", "D"]
        answer = f"({choice_list[example['answer']]})"
        return answer

    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["mmlu", example["subject"]],
            "source": "MMLU",
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU-Pro
elif args.dataset_name == "MMLU-Pro":
    load_dataset_path = args.dataset_path if args.dataset_path else "TIGER-Lab/MMLU-Pro"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def format_mmlu_query(example):
        query = "The following is a multiple-choice question:\n"
        query += example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        for idx, option in enumerate(example["options"]):
            query += f"\n({option_list[idx]}) {option}"
        return query

    def format_mmlu_gt(example):
        answer = f"The answer is ({option_list[example['answer_index']]}) {example['options'][example['answer_index']]}"
        return answer

    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["MMLU-Pro", example["category"], example["src"]],
            "source": "MMLU-Pro",
            "num_choices": len(example["options"]),
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    for i, item in enumerate(data_list):
        item["task_id"] = f"MMLU-Pro_{i:04d}"

# load GSM-Hard dataset
elif args.dataset_name == "GSM-Hard":
    load_dataset_path = args.dataset_path if args.dataset_path else "reasoning-machines/gsm-hard"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {"query": example["input"], "gt": str(example["target"]), "tag": ["math", "GSM-Hard"], "source": "GSM-Hard"}
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load GPQA dataset
elif args.dataset_name.startswith("GPQA"):
    load_dataset_path = args.dataset_path if args.dataset_path else "Idavidrein/gpqa"
    if args.dataset_name == "GPQA-Diamond":
        dataset = load_dataset(load_dataset_path, "gpqa_diamond", split="train", trust_remote_code=True)
    else:
        dataset = load_dataset(load_dataset_path, "gpqa_main", split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_gpqa_query(example):
        query = example["Question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['Correct Answer']}"
        query += f"\n(B) {example['Incorrect Answer 1']}"
        query += f"\n(C) {example['Incorrect Answer 2']}"
        query += f"\n(D) {example['Incorrect Answer 3']}"
        return query

    def format_gpqa_gt(example):
        answer = f"(A) {example['Correct Answer']}"
        return answer

    data_list = [
        {
            "query": format_gpqa_query(example),
            "gt": format_gpqa_gt(example),
            "tag": [
                args.dataset_name,
                example["High-level domain"],
                example["Subdomain"],
                example["Writer's Difficulty Estimate"],
            ],
            "source": args.dataset_name,
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load SciBench dataset
elif args.dataset_name == "SciBench":
    load_dataset_path = args.dataset_path if args.dataset_path else "xw27/scibench"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_scibench_gt(example):
        answer = f"{example['answer_number']}, the unit is {example['unit']}."
        return answer

    data_list = [
        {
            "query": example["problem_text"],
            "gt": format_scibench_gt(example),
            "tag": [args.dataset_name, "science", example["source"]],
            "source": args.dataset_name,
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

elif args.dataset_name == "AIME-2024":
    load_dataset_path = args.dataset_path if args.dataset_path else "Maxwell-Jia/AIME_2024"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["Problem"],
            "gt": example["Answer"],
            "tag": [args.dataset_name, "math"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]

elif args.dataset_name == "AIME-2025":
    load_dataset_path = args.dataset_path if args.dataset_path else "MathArena/aime_2025"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": str(example["answer"]),
            "tag": [args.dataset_name, "math"] + example["problem_type"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]

elif args.dataset_name == "HumanEval":
    load_dataset_path = args.dataset_path if args.dataset_path else "openai/openai_humaneval"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["prompt"],
            "gt": example["canonical_solution"],
            "test_cases": [example["test"]],
            "entry_point": example["entry_point"],
            "task_id": example["task_id"],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]
    # Also load MBPP sanitized variant
    dataset = load_dataset(load_dataset_path, "sanitized", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["prompt"] + "\n" + "\n".join(example["test_list"]),
            "gt": example["code"],
            "test_cases": example["test_list"],
            "task_id": str(example["task_id"]),
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]

elif args.dataset_name == "HumanEval+":
    # evalplus HumanEval+ (164 problems, full)
    load_dataset_path = args.dataset_path if args.dataset_path else "evalplus/humanevalplus"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["prompt"],
            "gt": example["canonical_solution"],
            "task_id": example["task_id"],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]

elif args.dataset_name == "MBPP+":
    # evalplus MBPP+ (~399 problems, full)
    load_dataset_path = args.dataset_path if args.dataset_path else "evalplus/mbppplus"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["prompt"],
            "gt": example["code"],
            "task_id": str(example["task_id"]),
            "test_cases": example["test_list"],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name,
        }
        for example in dataset
    ]

elif args.dataset_name == "APPS":
    # APPS test set (~5000 problems), sample 300 following prior work
    load_dataset_path = args.dataset_path if args.dataset_path else "codeparrot/apps"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question"],
            "gt": json.loads(example["solutions"])[0] if example["solutions"] else "",
            "input_output": example["input_output"],
            "difficulty": example["difficulty"],
            "starter_code": example.get("starter_code", ""),
            "task_id": str(example["problem_id"]),
            "tag": [args.dataset_name, "code", example["difficulty"]],
            "source": args.dataset_name,
        }
        for example in dataset
        if example["solutions"]  # skip problems without solutions
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

elif args.dataset_name == "LiveCodeBench":
    # LiveCodeBench code generation, hard only
    load_dataset_path = args.dataset_path if args.dataset_path else "livecodebench/code_generation_lite"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question_content"],
            "gt": "",  # no canonical solution provided, eval via test cases
            "input_output": example.get("public_test_cases", ""),
            "task_id": example["question_id"],
            "difficulty": example.get("difficulty", ""),
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name,
        }
        for example in dataset
        if example.get("difficulty", "").lower() == "hard"
    ]

elif args.dataset_name == "DS-1000":
    # DS-1000: 1000 data science problems (full)
    load_dataset_path = args.dataset_path if args.dataset_path else "xlangai/DS-1000"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["prompt"],
            "gt": example["reference_code"],
            "metadata": {
                "lib": example["metadata"]["library"],
                "perturbation_type": example["metadata"].get("perturbation_type", ""),
            },
            "task_id": str(idx),
            "tag": [args.dataset_name, "code", example["metadata"]["library"]],
            "source": args.dataset_name,
        }
        for idx, example in enumerate(dataset)
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

else:
    raise ValueError(f"Dataset {args.dataset_name} not supported.")


sample_pool = deduplicate(data_list)
print(f">> A data sample from the pool:\n{sample_pool[0]}")

print(f"{'='*50}\n There are {len(sample_pool)} queries in the pool.")

with open(save_path, "w") as output_json:
    json.dump(sample_pool, output_json, indent=4)
