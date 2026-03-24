import os
from transformers import AutoTokenizer
from KvChat import LLM, SamplingParams
import readline

def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=2,
    )
    temperature = 1.0
    max_tokens = 32000
    messages = []
    compress_enabled = True
    print("welcome to KvChat!")
    print("=== Streaming Chat CLI ===")
    print("Commands:")
    print("  /exit            exit")
    print("  /reset           clear history")
    print("  /compress on     enable compression")
    print("  /compress off    disable compression")
    print("")
    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye.")
            break
        if user_input == "/reset":
            messages = []
            print("Session reset.")
            continue
        if user_input == "/compress on":
            compress_enabled = True
            llm.model_runner.config.kv_compress_enabled = True
            print("Compression enabled.")
            continue
        if user_input == "/compress off":
            compress_enabled = False
            llm.model_runner.config.kv_compress_enabled = False
            print("Compression disabled.")
            continue
        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        llm.model_runner.config.kv_compress_enabled = compress_enabled
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm.add_request(prompt, sampling_params)
        print("Assistant: ", end="", flush=True)
        generated_token_ids = []
        while not llm.is_finished():
            outputs, num_tokens, step_token_ids = llm.step()
            if step_token_ids is not None:
                for tid in step_token_ids:
                    generated_token_ids.append(tid)
                    text = tokenizer.decode([tid], skip_special_tokens=False)
                    print(text, end="", flush=True)
        print()
        assistant_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
