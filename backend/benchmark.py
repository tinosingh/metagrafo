"""Module for benchmarking transcription performance."""

import time

from core.model_loader import ModelLoader


def benchmark(model_name):
    print(f"\n=== BENCHMARKING {model_name.upper()} ===")

    # Load model
    start = time.time()
    ml = ModelLoader(model_name)
    load_time = time.time() - start

    # Warmup
    _ = ml.transcribe("../frontend/punainen_linnake.mp3")

    # Timed inference
    start = time.time()
    result = ml.transcribe("../frontend/punainen_linnake.mp3")
    infer_time = time.time() - start

    return {
        "model": model_name,
        "load_time": f"{load_time:.2f}s",
        "inference_time": f"{infer_time:.2f}s",
        "text": result["text"][:100] + "...",
        "language": result.get("language", "en"),
    }


if __name__ == "__main__":
    models = ["tiny", "base", "small", "medium", "large-v3"]

    print("\n=== WHISPER BENCHMARK ===")
    for model in models:
        stats = benchmark(model)
        print(f"\n{stats['model']}:")
        print(f"- Load: {stats['load_time']}")
        print(f"- Infer: {stats['inference_time']}")
        print(f"- Language: {stats['language']}")
        print(f"- Sample: {stats['text']}")
