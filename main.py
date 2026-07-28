import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Cluster smoke test")
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("MENTAL_CHATBOT - CLUSTER SMOKE TEST")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Args received -> model_checkpoint={args.model_checkpoint}, dataset={args.dataset}")

    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("!! CUDA NOT AVAILABLE - GPU was not exposed to the container correctly !!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"Visible GPU(s): {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {torch.cuda.get_device_name(i)} - {props.total_memory / 1e9:.1f} GB")

    print("\nRunning a small matmul on GPU...")
    start = time.time()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Matmul OK - checksum: {c.sum().item():.4f} (took {elapsed:.3f}s)")

    try:
        import unsloth  # noqa: F401
        print("unsloth import OK")
    except Exception as e:
        print(f"unsloth import FAILED: {e}")

    print("\nSMOKE TEST PASSED - environment is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
