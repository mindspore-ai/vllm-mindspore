import vllm_mindspore # Add this line on the top of script.

from PIL import Image
from dataclasses import asdict

from vllm import LLM, SamplingParams


# Qwen2.5-VL
def get_llm(model_path: str, question: str, modality: str):

    llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=True,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    
    return llm, prompt, stop_token_ids


def main(args):
    
    # Prepare args and inputs.
    img_question = "What is the content of this image?"
    img = Image.open("./imgs/1.jpg").convert("RGB")
    llm, prompt, stop_token_ids = get_llm(args.model_path, img_question, "image")
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": img
        },
    } for _ in range(2)]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)
    
    # Run generate
    outputs = llm.generate(inputs, sampling_params)
    
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    args, _ = parser.parse_known_args()

    main(args)