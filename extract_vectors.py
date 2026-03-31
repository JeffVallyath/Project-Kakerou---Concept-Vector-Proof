"""Concept Vector Extraction — the core mech interp experiment.

Implements the methodology from:
- Zou et al. (2023) "Representation Engineering" — linear probing on
  population-level representations to extract concept directions
- Burns et al. (2022) "Eliciting Latent Knowledge" — finding internal
  belief directions independent of model output

Pipeline:
1. Run contrast pairs through local model
2. Cache residual stream activations at target layers
3. Compute difference vectors (positive - negative activation)
4. Extract concept direction via PCA on difference vectors
5. Train linear probe to validate the direction
6. Export concept vectors as .pt files

The resulting concept vectors can replace prompt-based signal extraction
in Kakerou's Layer 2, providing deterministic, high-resolution signals.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from dataset import build_full_dataset, ContrastPair
from dataset_extended import build_extended_full_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B"
TARGET_LAYERS = list(range(10, 20))        # mid-to-late layers (where concepts live)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("vectors")


# ---------------------------------------------------------------------------
# Step 1: Extract residual stream activations using nnsight
# ---------------------------------------------------------------------------

def extract_activations(
    model_name: str = MODEL_NAME,
    target_layers: list[int] = TARGET_LAYERS,
) -> dict[str, dict[str, Any]]:
    """Run all contrast pairs through the model and cache activations.

    For each pair, we extract the residual stream activation at the
    LAST TOKEN position for each target layer. Following Zou et al.,
    the last token's representation captures the model's "summary"
    understanding of the full input.

    Returns:
        {signal_name: {
            "positive_activations": [tensor per layer],
            "negative_activations": [tensor per layer],
            "pairs": [ContrastPair objects],
        }}
    """
    from nnsight import LanguageModel

    print(f"Loading model: {model_name}")
    print(f"Device: {DEVICE}")
    # Qwen3-4B at float16 fits in ~8GB VRAM (RTX 3080 = 10GB)
    model = LanguageModel(model_name, device_map=DEVICE, torch_dtype=torch.float16)

    dataset = build_extended_full_dataset()
    results = {}

    for signal_name, pairs in dataset.items():
        print(f"\n--- Extracting activations for: {signal_name} ({len(pairs)} pairs) ---")

        pos_activations = {layer: [] for layer in target_layers}
        neg_activations = {layer: [] for layer in target_layers}

        for i, pair in enumerate(pairs):
            if (i + 1) % 5 == 0:
                print(f"  Processing pair {i+1}/{len(pairs)}")

            # Extract positive (high-signal) activation
            pos_acts = {}
            with model.trace(pair.positive):
                for layer_idx in target_layers:
                    out = model.model.layers[layer_idx].output[0]
                    pos_acts[layer_idx] = out[-1, :].save()

            for layer_idx in target_layers:
                val = pos_acts[layer_idx]
                t = val.value if hasattr(val, 'value') else val
                pos_activations[layer_idx].append(t.detach().cpu())

            # Extract negative (low-signal) activation
            neg_acts = {}
            with model.trace(pair.negative):
                for layer_idx in target_layers:
                    out = model.model.layers[layer_idx].output[0]
                    neg_acts[layer_idx] = out[-1, :].save()

            for layer_idx in target_layers:
                val = neg_acts[layer_idx]
                t = val.value if hasattr(val, 'value') else val
                neg_activations[layer_idx].append(t.detach().cpu())

        # Stack into tensors
        results[signal_name] = {
            "positive_activations": {
                layer: torch.stack(acts) for layer, acts in pos_activations.items()
            },
            "negative_activations": {
                layer: torch.stack(acts) for layer, acts in neg_activations.items()
            },
            "pairs": pairs,
            "n_pairs": len(pairs),
        }

        print(f"  Done. Shape per layer: {results[signal_name]['positive_activations'][target_layers[0]].shape}")

    return results


# ---------------------------------------------------------------------------
# Step 2: Extract concept vectors via RepE methodology
# ---------------------------------------------------------------------------

def extract_concept_vectors(
    activation_data: dict[str, dict[str, Any]],
    target_layers: list[int] = TARGET_LAYERS,
    method: str = "mean_diff",  # "mean_diff" (RepE) or "pca" (RepE variant) or "probe" (ELK-style)
) -> dict[str, dict[int, torch.Tensor]]:
    """Extract concept direction vectors from contrast pair activations.

    Three methods following the literature:

    1. mean_diff (Zou et al. primary method):
       concept_vector = mean(positive_acts) - mean(negative_acts)
       Simple, robust, works well when pairs are high quality.

    2. pca (Zou et al. variant):
       Compute PCA on the difference vectors (pos - neg per pair).
       First principal component = concept direction.
       Better when pairs have varying quality.

    3. probe (Burns et al. / ELK):
       Train logistic regression, use weight vector as direction.
       Most robust but requires train/test split.

    Returns:
        {signal_name: {layer_idx: concept_vector_tensor}}
    """
    concept_vectors = {}

    for signal_name, data in activation_data.items():
        print(f"\n--- Extracting concept vectors for: {signal_name} ---")
        concept_vectors[signal_name] = {}

        for layer_idx in target_layers:
            pos = data["positive_activations"][layer_idx].squeeze(1)  # (n_pairs, hidden_dim)
            neg = data["negative_activations"][layer_idx].squeeze(1)

            if method == "mean_diff":
                # Zou et al. primary: mean difference
                vec = (pos.mean(dim=0) - neg.mean(dim=0))
                vec = vec / vec.norm()  # normalize to unit vector

            elif method == "pca":
                # Zou et al. variant: PCA on difference vectors
                diffs = pos - neg  # (n_pairs, hidden_dim)
                diffs_centered = diffs - diffs.mean(dim=0)

                # SVD to get first principal component
                U, S, Vh = torch.linalg.svd(diffs_centered, full_matrices=False)
                vec = Vh[0]  # first right singular vector
                vec = vec / vec.norm()

            elif method == "probe":
                # Burns et al. / ELK: logistic regression weight vector
                X = torch.cat([pos, neg], dim=0).numpy()
                y = np.array([1] * len(pos) + [0] * len(neg))

                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X, y)

                vec = torch.tensor(clf.coef_[0], dtype=torch.float32)
                vec = vec / vec.norm()

            concept_vectors[signal_name][layer_idx] = vec

        # Find the best layer (highest separation)
        best_layer, best_sep = _find_best_layer(
            concept_vectors[signal_name], data, target_layers
        )
        print(f"  Best layer: {best_layer} (separation: {best_sep:.4f})")

    return concept_vectors


def _find_best_layer(
    vectors: dict[int, torch.Tensor],
    data: dict[str, Any],
    layers: list[int],
) -> tuple[int, float]:
    """Find the layer where the concept vector gives best separation."""
    best_layer = layers[0]
    best_sep = 0.0

    for layer in layers:
        vec = vectors[layer]
        pos = data["positive_activations"][layer].squeeze(1)
        neg = data["negative_activations"][layer].squeeze(1)

        # Project onto concept vector
        pos_proj = (pos @ vec).mean().item()
        neg_proj = (neg @ vec).mean().item()
        separation = abs(pos_proj - neg_proj)

        if separation > best_sep:
            best_sep = separation
            best_layer = layer

    return best_layer, best_sep


# ---------------------------------------------------------------------------
# Step 3: Validate with linear probe (quantitative evaluation)
# ---------------------------------------------------------------------------

def validate_probes(
    activation_data: dict[str, dict[str, Any]],
    target_layers: list[int] = TARGET_LAYERS,
    n_folds: int = 5,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate linear probes with stratified k-fold cross-validation.

    This validates that the concept IS linearly represented in the
    residual stream — a core claim of both Zou et al. and Burns et al.

    Uses 5-fold stratified CV instead of a single train/test split to
    produce robust accuracy estimates with standard deviations.
    """
    from sklearn.model_selection import StratifiedKFold

    results = {}

    for signal_name, data in activation_data.items():
        print(f"\n--- Validating probe for: {signal_name} ({n_folds}-fold CV) ---")
        results[signal_name] = {"layers": {}, "best_layer": 0, "best_accuracy": 0.0}

        for layer_idx in target_layers:
            pos = data["positive_activations"][layer_idx].squeeze(1).numpy()
            neg = data["negative_activations"][layer_idx].squeeze(1).numpy()

            X = np.vstack([pos, neg])
            y = np.array([1] * len(pos) + [0] * len(neg))

            # Stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_train_accs = []
            fold_test_accs = []

            for train_idx, test_idx in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X[train_idx], y[train_idx])
                fold_train_accs.append(clf.score(X[train_idx], y[train_idx]))
                fold_test_accs.append(clf.score(X[test_idx], y[test_idx]))

            mean_train = float(np.mean(fold_train_accs))
            mean_test = float(np.mean(fold_test_accs))
            std_test = float(np.std(fold_test_accs))

            results[signal_name]["layers"][layer_idx] = {
                "train_accuracy": round(mean_train, 4),
                "test_accuracy": round(mean_test, 4),
                "test_std": round(std_test, 4),
                "fold_accuracies": [round(a, 4) for a in fold_test_accs],
                "n_samples": len(X),
            }

            if mean_test > results[signal_name]["best_accuracy"]:
                results[signal_name]["best_accuracy"] = mean_test
                results[signal_name]["best_accuracy_std"] = std_test
                results[signal_name]["best_layer"] = layer_idx

        best = results[signal_name]
        print(f"  Best layer: {best['best_layer']} — "
              f"test accuracy: {best['best_accuracy']:.1%} "
              f"(±{best.get('best_accuracy_std', 0):.1%})")
        # Show per-layer summary
        for layer_idx in target_layers:
            lr = results[signal_name]["layers"][layer_idx]
            print(f"    Layer {layer_idx}: {lr['test_accuracy']:.1%} ±{lr['test_std']:.1%} "
                  f"(train: {lr['train_accuracy']:.1%})")

    return results


# ---------------------------------------------------------------------------
# Step 4: Save concept vectors for Kakerou integration
# ---------------------------------------------------------------------------

def save_vectors(
    concept_vectors: dict[str, dict[int, torch.Tensor]],
    validation_results: dict[str, dict[str, Any]],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save concept vectors and metadata for Kakerou integration.

    For each signal, saves the vector from the best-performing layer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for signal_name, layer_vectors in concept_vectors.items():
        best_layer = validation_results[signal_name]["best_layer"]
        best_acc = validation_results[signal_name]["best_accuracy"]
        vec = layer_vectors[best_layer]

        # Save vector
        vec_path = output_dir / f"{signal_name}_vector.pt"
        torch.save(vec, vec_path)

        manifest[signal_name] = {
            "vector_file": str(vec_path),
            "best_layer": best_layer,
            "test_accuracy": best_acc,
            "hidden_dim": vec.shape[0],
            "method": "mean_diff",
        }

        print(f"  Saved {signal_name}: layer {best_layer}, accuracy {best_acc:.1%}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_full_experiment(
    model_name: str = MODEL_NAME,
    method: str = "mean_diff",
) -> None:
    """Run the complete concept vector extraction pipeline.

    1. Extract activations from contrast pairs
    2. Compute concept vectors (RepE methodology)
    3. Validate with linear probes
    4. Save vectors for Kakerou integration
    """
    print("=" * 70)
    print("MECHANISTIC BEHAVIORAL SIGNAL EXTRACTION")
    print(f"Model: {model_name}")
    print(f"Method: {method}")
    print(f"Target layers: {TARGET_LAYERS}")
    print("=" * 70)

    t0 = time.time()

    # Step 1: Extract activations
    print("\n[STEP 1] Extracting residual stream activations...")
    activation_data = extract_activations(model_name, TARGET_LAYERS)

    # Step 2: Extract concept vectors
    print("\n[STEP 2] Computing concept vectors...")
    concept_vectors = extract_concept_vectors(activation_data, TARGET_LAYERS, method)

    # Step 3: Validate
    print("\n[STEP 3] Validating with linear probes...")
    validation_results = validate_probes(activation_data, TARGET_LAYERS)

    # Step 4: Save
    print("\n[STEP 4] Saving concept vectors...")
    save_vectors(concept_vectors, validation_results)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Experiment complete in {elapsed:.0f}s")
    print(f"{'=' * 70}")

    # Summary
    print("\nResults Summary:")
    for signal, result in validation_results.items():
        print(f"  {signal}: layer {result['best_layer']}, accuracy {result['best_accuracy']:.1%}")


if __name__ == "__main__":
    run_full_experiment()
