import os
from nanokvllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
def get_problem_list_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" in obj:
                problems.append(obj["problem"])
    return problems

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager = False, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=2000)
    prompts = [
        "You are given a math problem.\n\nProblem: Let \\(O=(0,0)\\), \\(A=\\left(\\tfrac{1}{2},0\\right)\\), and \\(B=\\left(0,\\tfrac{\\sqrt{3}}{2}\\right)\\) be points in the coordinate plane. Let \\(\\mathcal{F}\\) be the family of segments \\(\\overline{PQ}\\) of unit length lying in the first quadrant with \\(P\\) on the \\(x\\)-axis and \\(Q\\) on the \\(y\\)-axis. There is a unique point \\(C\\) on \\(\\overline{AB}\\), distinct from \\(A\\) and \\(B\\),  that does not belong to any segment from \\(\\mathcal{F}\\) other than \\(\\overline{AB}\\). Then \\(OC^2=\\tfrac{p}{q}\\), where \\(p\\) and \\(q\\) are relatively prime positive integers. Find \\(p+q\\).\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{}",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params,use_tqdm = True)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
