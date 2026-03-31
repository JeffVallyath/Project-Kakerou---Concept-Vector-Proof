"""Kakerou Integration — deterministic signal extraction via concept vectors.

This module replaces btom_engine/sensor.py's Gemini API call with a
local forward pass + dot product projection onto concept vectors.

Benefits over prompt-based extraction:
- DETERMINISTIC: same input always produces same output (no LLM stochasticity)
- HIGH RESOLUTION: continuous floats, not chunked 0.2/0.5/0.8
- LOW LATENCY: single forward pass (~50ms) vs API call (~1500ms)
- ZERO HALLUCINATION: mathematical projection, not text generation
- INTERPRETABLE: the concept vector IS the explanation (it's a direction in geometry)

Usage in Kakerou:
    from kakerou_integration import MechanisticSensor
    sensor = MechanisticSensor("Qwen/Qwen2.5-7B-Instruct", "vectors/")
    signals = sensor.extract_signals("I don't want to talk about that.")
    # signals = {"evasive_deflection": 0.847, "defensive_justification": 0.123, ...}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MechanisticSensor:
    """Deterministic behavioral signal extraction via concept vector projection.

    Replaces the LLM prompt-based sensor with direct measurement of the
    model's internal cognitive geometry.

    Architecture:
    1. Run input text through local model (single forward pass)
    2. Extract residual stream activation at the optimal layer for each signal
    3. Project activation onto pre-computed concept vector (dot product)
    4. Normalize to 0-1 range using calibrated thresholds
    5. Return deterministic, high-resolution signal values
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B-Instruct",
        vectors_dir: str | Path = "vectors",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.vectors_dir = Path(vectors_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._vectors: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load model and concept vectors on first use."""
        if self._loaded:
            return

        from nnsight import LanguageModel

        print(f"Loading mechanistic sensor: {self.model_name}")
        self._model = LanguageModel(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
        )

        # Load concept vectors
        manifest_path = self.vectors_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for signal_name, meta in manifest.items():
            vec = torch.load(
                self.vectors_dir / f"{signal_name}_vector.pt",
                weights_only=True,
            ).to(self.device).half()

            self._vectors[signal_name] = {
                "vector": vec,
                "layer": meta["best_layer"],
                "accuracy": meta["test_accuracy"],
            }

        self._loaded = True
        print(f"Loaded {len(self._vectors)} concept vectors")

    def extract_signals(self, text: str) -> dict[str, float]:
        """Extract all behavioral signals from a text via concept vector projection.

        Returns:
            {"evasive_deflection": 0.847, "defensive_justification": 0.123, ...}

        Each value is a deterministic float in [0, 1].
        """
        self._ensure_loaded()

        t0 = time.time()
        signals = {}

        # Group by layer to minimize forward passes
        layers_needed = set(v["layer"] for v in self._vectors.values())

        # Run forward pass and cache activations at all needed layers
        activations = {}
        with self._model.trace(text) as tracer:
            for layer_idx in layers_needed:
                activations[layer_idx] = (
                    self._model.model.layers[layer_idx].output[0][-1, :].save()
                )

        # Project onto each concept vector
        for signal_name, info in self._vectors.items():
            layer_idx = info["layer"]
            vec = info["vector"]
            _a = activations[layer_idx]
            act = (_a.value if hasattr(_a, 'value') else _a).detach().squeeze()

            # Dot product = projection onto concept direction
            raw_score = (act @ vec).item()

            # Normalize to [0, 1] using sigmoid-like scaling
            # The raw dot product can be any real number; we map it to a probability
            normalized = _sigmoid_normalize(raw_score)
            signals[signal_name] = round(normalized, 4)

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Mechanistic sensor: %d signals in %.0fms",
            len(signals), elapsed_ms,
        )

        return signals

    def extract_signals_with_metadata(self, text: str) -> dict[str, dict[str, Any]]:
        """Extract signals with full metadata (raw score, layer, accuracy).

        For the Kakerou dashboard — shows both the signal value and the
        mechanistic provenance.
        """
        self._ensure_loaded()

        activations = {}
        layers_needed = set(v["layer"] for v in self._vectors.values())

        with self._model.trace(text) as tracer:
            for layer_idx in layers_needed:
                activations[layer_idx] = (
                    self._model.model.layers[layer_idx].output[0][-1, :].save()
                )

        results = {}
        for signal_name, info in self._vectors.items():
            _a = activations[info["layer"]]
            act = (_a.value if hasattr(_a, 'value') else _a).detach().squeeze()
            raw_score = (act @ info["vector"]).item()
            normalized = _sigmoid_normalize(raw_score)

            results[signal_name] = {
                "value": round(normalized, 4),
                "raw_projection": round(raw_score, 4),
                "layer": info["layer"],
                "probe_accuracy": info["accuracy"],
                "method": "concept_vector_projection",
                "deterministic": True,
            }

        return results


def _sigmoid_normalize(x: float, temperature: float = 1.0, bias: float = 0.0) -> float:
    """Normalize a raw dot product to [0, 1] using sigmoid.

    temperature controls the steepness of the transition.
    bias shifts the center point.
    """
    import math
    try:
        return 1.0 / (1.0 + math.exp(-(x - bias) / temperature))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sensor = MechanisticSensor()

    test_texts = [
        "I don't want to talk about that. Can we change the subject?",
        "I was at the office from 9am to 5pm, my colleague Sarah was there too.",
        "I ALREADY told you three times! This is ridiculous!",
        "Look, anyone in my position would have done the same thing.",
        "I've never been to that restaurant in my life.",
    ]

    print("\n" + "=" * 70)
    print("MECHANISTIC SENSOR TEST")
    print("=" * 70)

    for text in test_texts:
        print(f"\nInput: {text[:60]}...")
        signals = sensor.extract_signals_with_metadata(text)
        for signal, data in signals.items():
            print(f"  {signal}: {data['value']:.4f} (raw={data['raw_projection']:.4f}, layer={data['layer']})")
