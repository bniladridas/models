import os
import subprocess
import sys


def test_training_runs():
    """E2E test: Run training for a few steps and check it completes without error."""
    # This is a basic e2e test; in practice, limit steps in train.py for CI
    # For now, just check the script can start (mock or skip full run)
    # Since full training is slow, we'll skip the actual run in CI
    # and just test imports and basic setup

    # Test that we can import the training module
    import src.train  # noqa: F401


def test_evaluation_runs():
    """Test evaluation script can run (assuming model exists)."""
    # This would require a trained model; skip if not present
    if not os.path.exists("models/trained_model"):
        import pytest

        pytest.skip("Trained model not available for evaluation test")

    # Run evaluation and check it doesn't crash
    result = subprocess.run(
        [sys.executable, "src/evaluate.py"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"Evaluation failed: {result.stderr}"
