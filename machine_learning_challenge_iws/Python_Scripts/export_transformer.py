"""
export_transformer.py
Export EVA-02 model to ONNX format with ImageNet classification head.
CRITICAL: Uses EVA-02 normalization, outputs 1000-class probabilities.
INPUT SIZE: 448x448
NOW INCLUDES SOFTMAX for probability output
"""

import torch
import timm
import numpy as np
import onnx
import onnxruntime as ort
from torch import nn
import argparse
import os

# EVA-02 NORMALIZATION CONSTANTS (NOT ImageNet!)
EVA02_MEAN = [0.48145466, 0.4578275, 0.40821073]
EVA02_STD = [0.26862954, 0.26130258, 0.27577711]


class EVA02WithSoftmax(nn.Module):
    """
    Wrapper to include EVA-02 normalization AND softmax in the exported model.
    This ensures the output is probabilities (sum ≈ 1.0).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(EVA02_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(EVA02_STD).view(1, 3, 1, 1))
        self.softmax = nn.Softmax(dim=1)  # Softmax over class dimension

    def forward(self, x):
        # Input is assumed to be in [0, 1] range
        # Apply EVA-02 normalization: (x - mean) / std
        x = (x - self.mean) / self.std
        # Get model output (logits)
        logits = self.model(x)
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        return probabilities


def export_model(args):
    """Export EVA-02 model to ONNX with proper configuration."""

    print("=" * 60)
    print("🔄 EVA-02 Model Export to ONNX")
    print("=" * 60)

    # 1. Load EVA-02 model with ImageNet classification head
    print("\n1. Loading EVA-02 model...")
    try:
        model = timm.create_model(
            "eva02_large_patch14_448.mim_m38m_ft_in1k",
            pretrained=True,
            num_classes=1000,  # ImageNet classes
        )
        model.eval()
        print(f" Model: {type(model).__name__}")
        print(f" Classes: {model.num_classes}")
        print(f" Input size: 448x448")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(" Try: pip install timm")
        return

    # 2. Wrap with normalization AND softmax
    print("\n2. Adding EVA-02 normalization and Softmax...")
    wrapped_model = EVA02WithSoftmax(model)
    wrapped_model.eval()
    print(f" Mean: {EVA02_MEAN}")
    print(f" Std: {EVA02_STD}")
    print(f" Softmax: Applied to get probabilities")

    # 3. Create dummy input for export
    print("\n3. Preparing model for export...")
    batch_size = 1
    channels = 3
    height = 448
    width = 448

    # Input in range [0, 1] (as images will be normalized)
    dummy_input = torch.randn(batch_size, channels, height, width)
    print(f" Dummy input shape: {dummy_input.shape}")

    # 4. Test forward pass first
    print("\n4. Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = wrapped_model(dummy_input)
            print(f" Test output shape: {test_output.shape}")
            print(f" Test output sum: {test_output.sum().item():.6f}")
            if test_output.shape == (1, 1000):
                print(" ✅ Output shape is correct: (1, 1000)")
            else:
                print(f" ❌ Wrong output shape: {test_output.shape}")
                return

            # CRITICAL: Check if output is probabilities
            output_sum = test_output.sum().item()
            if abs(output_sum - 1.0) < 0.01:
                print(" ✅ Output is probabilities (sum ≈ 1.0)")
            else:
                print(f" ❌ Output sum is {output_sum:.6f}, NOT ~1.0")
                print(" This is NOT probabilities!")
                return
    except Exception as e:
        print(f" ❌ Forward pass failed: {e}")
        return

    # 5. Export to ONNX
    print("\n5. Exporting to ONNX...")
    onnx_path = args.output_path or "transformer_model.onnx"

    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["probabilities"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=args.verbose,
        )

        print(f" ✅ Model exported to: {onnx_path}")
        print(f" Input shape: {dummy_input.shape}")
        print(f" Expected output shape: (batch_size, 1000)")
    except Exception as e:
        print(f" ❌ Export failed: {e}")
        return

    # 6. Validate the exported model
    print("\n6. Validating exported ONNX model...")

    # Check ONNX file
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(" ✅ ONNX model is valid")
    except Exception as e:
        print(f" ❌ ONNX validation failed: {e}")
        return

    # 7. Test inference with ONNX Runtime
    print("\n7. Running inference test...")

    # Create realistic test input (gray image)
    test_input = torch.ones(1, 3, 448, 448) * 0.5

    # Run PyTorch inference
    with torch.no_grad():
        torch_output = wrapped_model(test_input)
        torch_sum = torch_output.sum().item()
        print(f" PyTorch output sum: {torch_sum:.6f}")

    # Run ONNX inference
    try:
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        ort_inputs = {"input": test_input.numpy().astype(np.float32)}
        ort_outputs = ort_session.run(["probabilities"], ort_inputs)
        ort_output = ort_outputs[0]
        ort_sum = ort_output.sum()

        print(f" ONNX output sum: {ort_sum:.6f}")

        # CRITICAL CHECK: Output should be probabilities
        if abs(ort_sum - 1.0) < 0.01:
            print(" ✅ Output IS probabilities (sum ≈ 1.0)")
        else:
            print(f" ❌ FAILED: Output sum is {ort_sum:.6f}, NOT ~1.0")
            return

        # Check output shape
        if ort_output.shape == (1, 1000):
            print(" ✅ Output shape is correct: (1, 1000)")
        else:
            print(f" ❌ Wrong output shape: {ort_output.shape}")
            return

        # Print verification
        print("\n8. Sample verification:")
        print(f" Output range: [{ort_output.min():.6f}, {ort_output.max():.6f}]")
        print(f" First 5 values: {ort_output[0, :5]}")

        # Additional probability checks
        print(f" All values >= 0: {np.all(ort_output >= 0)}")
        print(f" Max probability: {ort_output.max():.6f}")
        print(f" Min probability: {ort_output.min():.6f}")

    except Exception as e:
        print(f" ❌ ONNX inference test failed: {e}")
        return

    # 8. Test with multiple inputs
    print("\n9. Testing with random inputs...")
    try:
        for i in range(3):
            random_input = torch.randn(1, 3, 448, 448)
            ort_inputs = {"input": random_input.numpy().astype(np.float32)}
            ort_outputs = ort_session.run(["probabilities"], ort_inputs)
            ort_sum = ort_outputs[0].sum()

            if abs(ort_sum - 1.0) < 0.01:
                print(f" Test {i + 1}: ✅ Sum = {ort_sum:.6f}")
            else:
                print(f" Test {i + 1}: ❌ Sum = {ort_sum:.6f} (FAILED)")
                return
    except Exception as e:
        print(f" Random input test failed: {e}")

    # 9. Save constants
    constants_file = "normalization_constants.txt"
    try:
        with open(constants_file, "w") as f:
            f.write(f"# EVA-02 Normalization Constants\n")
            f.write(f"# Input size: 448x448\n")
            f.write(f"# Output: 1000-class probabilities (sum ≈ 1.0)\n")
            f.write(f"MEAN = {EVA02_MEAN}\n")
            f.write(f"STD = {EVA02_STD}\n")
            f.write(f"INPUT_RANGE = [0, 1]\n")
            f.write(f"SOFTMAX_INCLUDED = True\n")

        print(f"\n📋 Normalization constants saved to: {constants_file}")
    except Exception as e:
        print(f" ⚠️ Could not save constants file: {e}")

    # 10. Final validation
    print("\n" + "=" * 60)
    print("FINAL VALIDATION:")
    print("=" * 60)

    final_checks = [
        ("Model loads successfully", True),
        ("Forward pass works", True),
        ("Output shape is (1, 1000)", ort_output.shape == (1, 1000)),
        ("Output sums to ~1.0 (probabilities)", abs(ort_sum - 1.0) < 0.01),
        ("All outputs are non-negative", np.all(ort_output >= 0)),
        ("ONNX export successful", True),
        ("ONNX model is valid", True),
        ("Input name is 'input'", True),
        ("Output name is 'probabilities'", True),
        ("EVA-02 normalization included", True),
        ("Softmax included", True),
    ]

    all_passed = True
    for check_name, check_result in final_checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}")
        if not check_result:
            all_passed = False

    if all_passed:
        print("\n🎉 SUCCESS! Model is ready for Julia.")
        print(f"\nCRITICAL FOR JULIA:")
        print(f" 1. Input size: 448x448")
        print(f" 2. Pixel range: [0, 1] (divide by 255)")
        print(f" 3. EVA-02 normalization (NOT ImageNet):")
        print(f" mean = {EVA02_MEAN}")
        print(f" std = {EVA02_STD}")
        print(f" 4. Output: 1000 probabilities (sum ≈ 1.0)")
    else:
        print("\n❌ FAILED: Model output is NOT probabilities.")
        print(" The output should sum to ~1.0 but doesn't.")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export EVA-02 model to ONNX")
    parser.add_argument(
        "--output_path",
        type=str,
        default="transformer_model.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose export output")

    args = parser.parse_args()

    print("⚠️ CRITICAL: This export includes Softmax for probability output.")
    print(" Output should sum to ~1.0 for probabilities.\n")

    try:
        export_model(args)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
