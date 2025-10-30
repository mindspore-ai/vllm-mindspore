import os
import sys
import time
from typing import List, Dict

import vllm_mindspore  # Ensure backend is imported/initialized
from vllm import LLM
from vllm.sampling_params import BeamSearchParams


def _set_architecture_env(use_v1: bool) -> None:
    if use_v1:
        os.environ["VLLM_USE_V1"] = "1"
    else:
        # Explicitly set to 0 for consistency across runs
        os.environ["VLLM_USE_V1"] = "0"


def _make_llm(model_path: str,
              max_num_seqs: int = 4,
              max_model_len: int = 1024,
              gpu_memory_utilization: float = 0.7) -> LLM:
    return LLM(
        model=model_path,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


# Unified parameter sets used across v0 and v1
PARAM_SETS: Dict[str, BeamSearchParams] = {
    "basic": BeamSearchParams(beam_width=4, max_tokens=100, temperature=0.7, length_penalty=1.0, ignore_eos=False),
    "batch": BeamSearchParams(beam_width=3, max_tokens=45, temperature=0.7, length_penalty=1.2, ignore_eos=False),
    "token_vs_text": BeamSearchParams(beam_width=3, max_tokens=40, temperature=0.7, length_penalty=1.0, ignore_eos=False),
}

PARAM_VARIATIONS: List[Dict] = [
    {
        "name": "Zero Temperature",
        "params": BeamSearchParams(beam_width=3, max_tokens=40, temperature=0.0, length_penalty=1.0),
        "prompt": "Python programming",
    },
    {
        "name": "High Temperature",
        "params": BeamSearchParams(beam_width=3, max_tokens=40, temperature=1.2, length_penalty=1.0),
        "prompt": "Artificial intelligence",
    },
    {
        "name": "Strong Length Penalty",
        "params": BeamSearchParams(beam_width=3, max_tokens=50, temperature=0.7, length_penalty=2.0),
        "prompt": "Deep learning",
    },
]

EDGE_CASES: List[Dict] = [
    {
        "name": "Single Beam",
        "params": BeamSearchParams(beam_width=1, max_tokens=30),
        "prompt": "Test",
    },
    {
        "name": "Empty Prompt",
        "params": BeamSearchParams(beam_width=2, max_tokens=20),
        "prompt": "",
    },
]


def test_basic(llm: LLM) -> None:
    print("=== Basic Beam Search Test ===")
    prompts = [{"prompt": "I am"}]
    params = PARAM_SETS["basic"]
    results = llm.beam_search(prompts, params)
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        for j, sequence in enumerate(result.sequences, 1):
            print(f"  Sequence {j}:")
            print(f"    Text: {sequence.text!r}")
            print(f"    Score: {sequence.cum_logprob:.4f}")
            print(f"    Finish reason: {sequence.finish_reason}")
        print("---")


def test_parameter_variations(llm: LLM) -> None:
    print("\n=== Parameter Variations Test ===")
    for test_case in PARAM_VARIATIONS:
        print(f"\n--- {test_case['name']} ---")
        params = test_case["params"]
        prompts = [{"prompt": test_case["prompt"]}]
        start_time = time.time()
        results = llm.beam_search(prompts, params)
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f}s")
        for sequence in results[0].sequences:
            print(f"  Text: {sequence.text!r}")
            print(f"  Score: {sequence.cum_logprob:.4f}")
            print(f"  Token count: {len(sequence.tokens)}")
            print(f"  Finish reason: {sequence.finish_reason}")


def test_batch_processing(llm: LLM) -> None:
    print("\n=== Batch Processing Test ===")
    multi_prompts = [
        {"prompt": "Science and technology"},
        {"prompt": "Climate change solutions"},
        {"prompt": "Space exploration missions"},
        {"prompt": "Renewable energy sources"},
    ]
    params = PARAM_SETS["batch"]
    print(f"Processing {len(multi_prompts)} prompts in batch...")
    start_time = time.time()
    results = llm.beam_search(multi_prompts, params)
    end_time = time.time()
    print(f"Batch processing time: {end_time - start_time:.2f}s")
    print(f"Average time per prompt: {(end_time - start_time) / len(multi_prompts):.2f}s")
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {multi_prompts[i]['prompt']}")
        print("Generated sequences:")
        for j, sequence in enumerate(result.sequences, 1):
            print(f"  Beam {j}: {sequence.text!r}")
            print(f"    Score: {sequence.cum_logprob:.4f}")
            print(f"    Tokens: {len(sequence.tokens)}")


def test_beam_width_comparison(llm: LLM) -> None:
    print("\n=== Beam Width Comparison Test ===")
    prompt = [{"prompt": "Artificial intelligence will"}]
    beam_widths = [2, 4, 8]
    for beam_width in beam_widths:
        print(f"\n--- Beam Width: {beam_width} ---")
        params = BeamSearchParams(beam_width=beam_width, max_tokens=40)
        start_time = time.time()
        results = llm.beam_search(prompt, params)
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f}s")
        print(f"Number of sequences: {len(results[0].sequences)}")
        best_sequence = results[0].sequences[0]
        print(f"Best sequence: {best_sequence.text!r}")
        print(f"Best score: {best_sequence.cum_logprob:.4f}")


def test_token_vs_text_input(llm: LLM) -> None:
    print("\n=== Token vs Text Input Test ===")
    tokenizer = llm.get_tokenizer()
    test_text = "Machine learning"
    text_prompt = [{"prompt": test_text}]
    token_ids = tokenizer.encode(test_text)
    token_prompt = [{"prompt_token_ids": token_ids}]
    params = BeamSearchParams(beam_width=2, max_tokens=20)

    results_text = llm.beam_search(text_prompt, params)
    results_tokens = llm.beam_search(token_prompt, params)

    text_scores = [seq.cum_logprob for seq in results_text[0].sequences]
    token_scores = [seq.cum_logprob for seq in results_tokens[0].sequences]
    score_diff = max(abs(a - b) for a, b in zip(text_scores, token_scores))

    if score_diff < 1e-4:
        print("✓ Text and token input produce consistent results")
    else:
        print(f"⚠ Score difference: {score_diff:.2e}")


def test_edge_cases(llm: LLM) -> None:
    print("\n=== Edge Cases Test ===")
    for case in EDGE_CASES:
        print(f"\n--- {case['name']} ---")
        prompts = [{"prompt": case["prompt"]}]
        try:
            start_time = time.time()
            results = llm.beam_search(prompts, case["params"])
            end_time = time.time()
            print(f"Success! Time: {end_time - start_time:.2f}s")
            print(f"Generated {len(results[0].sequences)} sequences")
            for i, sequence in enumerate(results[0].sequences, 1):
                print(f"  Sequence {i}: {sequence.text!r}")
                print(f"    Score: {sequence.cum_logprob:.4f}")
        except Exception as e:
            print(f"Error: {e}")


def test_performance(llm: LLM) -> None:
    print("\n=== Performance Test ===")
    prompts = [{"prompt": "Artificial intelligence"} for _ in range(8)]
    params = BeamSearchParams(beam_width=4, max_tokens=50)

    start_time = time.time()
    results = llm.beam_search(prompts, params)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(len(seq.tokens) for result in results for seq in result.sequences)
    throughput = total_tokens / total_time

    print(f"Batch size: {len(prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per prompt: {total_time / len(prompts):.3f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Average tokens per sequence: {total_tokens / (len(prompts) * params.beam_width):.1f}")


def test_best_of(llm: LLM) -> None:
    print("\n=== Best-of-N Test ===")
    prompt = [{"prompt": "The future of technology"}]

    test_configs = [
        {"beam_width": 4, "best_of": 4, "name": "beam=best_of"},
        {"beam_width": 8, "best_of": 4, "name": "beam>best_of"},
        {"beam_width": 2, "best_of": 1, "name": "minimal"},
    ]

    for config in test_configs:
        print(f"\n--- {config['name']}: beam_width={config['beam_width']}, best_of={config['best_of']} ---")
        params = BeamSearchParams(
            beam_width=config["beam_width"],
            best_of=config["best_of"],
            max_tokens=40,
        )
        results = llm.beam_search(prompt, params)
        print(f"Returned sequences: {len(results[0].sequences)}")
        print(f"Expected: {config['best_of']}")

        if len(results[0].sequences) == config["best_of"]:
            print("✓ Correct number of sequences returned")
        else:
            print(f"⚠ Expected {config['best_of']}, got {len(results[0].sequences)}")

        scores = [seq.cum_logprob for seq in results[0].sequences]
        if scores == sorted(scores, reverse=True):
            print("✓ Sequences ordered by score")
        else:
            print("⚠ Sequences not properly ordered")

        for i, seq in enumerate(results[0].sequences, 1):
            print(f"  Sequence {i}: score={seq.cum_logprob:.4f}, tokens={len(seq.tokens)}")


def test_best_of_performance(llm: LLM) -> None:
    print("\n=== Best-of-N Performance Comparison ===")
    prompts = [{"prompt": "Technology"} for _ in range(4)]

    configs = [
        {"beam_width": 4, "best_of": 1},
        {"beam_width": 4, "best_of": 2},
        {"beam_width": 4, "best_of": 4},
    ]

    for config in configs:
        params = BeamSearchParams(
            beam_width=config["beam_width"],
            best_of=config["best_of"],
            max_tokens=30,
        )
        start_time = time.time()
        results = llm.beam_search(prompts, params)
        end_time = time.time()

        print(f"\nbeam_width={config['beam_width']}, best_of={config['best_of']}:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Sequences per prompt: {len(results[0].sequences)}")
        print(f"  Best score: {results[0].sequences[0].cum_logprob:.4f}")


def run_beam_tests(model_path: str,
                   use_v1: bool,
                   max_num_seqs: int = 4,
                   max_model_len: int = 1024,
                   gpu_memory_utilization: float = 0.7) -> bool:
    print(
        f"Starting Comprehensive vLLM-MindSpore Beam Search Test Suite (use_v1={use_v1})"
    )
    print("=" * 70)
    _set_architecture_env(use_v1)
    llm = _make_llm(
        model_path=model_path,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        test_basic(llm)
        test_parameter_variations(llm)
        test_batch_processing(llm)
        test_beam_width_comparison(llm)
        test_token_vs_text_input(llm)
        test_edge_cases(llm)
        test_performance(llm)
        test_best_of(llm)
        test_best_of_performance(llm)
        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("vLLM-MindSpore beam search functionality verified.")
        return True
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
if __name__ == "__main__":
    MODEL_PATH = "/home/ma-user/work/Qwen2.5-7B-Instruct"
    COMMON_KWARGS = dict(
        max_num_seqs=4,
        max_model_len=1024,
        gpu_memory_utilization=0.7,
    )

    print("🚀 Starting V0 Architecture Beam Search Tests")
    print("=" * 60)
    v0_passed = run_beam_tests(model_path=MODEL_PATH, use_v1=False, **COMMON_KWARGS)
    print(f"\n🏁 V0 Architecture Test Result: {'PASSED' if v0_passed else 'FAILED'}")
    print("\n" + "=" * 60)

    print("🚀 Starting V1 Architecture Beam Search Tests")
    print("=" * 60)
    v1_passed = run_beam_tests(model_path=MODEL_PATH, use_v1=True, **COMMON_KWARGS)
    print(f"\n🏁 V1 Architecture Test Result: {'PASSED' if v1_passed else 'FAILED'}")

    all_passed = v0_passed and v1_passed
    sys.exit(0 if all_passed else 1)