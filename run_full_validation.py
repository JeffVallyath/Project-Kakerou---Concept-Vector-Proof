"""Full Validation Pipeline — runs all 5 validation tasks.

1. Extract activations from extended dataset (261 pairs)
2. Compute concept vectors (RepE mean_diff)
3. 5-fold CV probe validation
4. Out-of-distribution test
5. Activation steering proof
6. Save results

Run: python run_full_validation.py
Requires: GPU with ~8GB VRAM, nnsight, torch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from extract_vectors import (
    extract_activations,
    extract_concept_vectors,
    validate_probes,
    save_vectors,
    TARGET_LAYERS,
    MODEL_NAME,
)
from dataset_extended import build_extended_full_dataset


# ---------------------------------------------------------------------------
# Out-of-distribution test pairs — text the model has NEVER seen
# These are intentionally different in style from the training pairs
# ---------------------------------------------------------------------------

OOD_EVASION = [
    # Different register/style than training data
    ("Per my last email, I believe this was addressed.", True),
    ("The quarterly results are in appendix B, pages 12-15.", False),
    ("I think that's more of a question for the board.", True),
    ("We shipped 14,000 units to the warehouse in Ohio on March 3rd.", False),
    ("Hmm that rings a bell but I couldn't say for certain.", True),
    ("I wrote the patch, tested it locally, and deployed at 5pm EST.", False),
    ("You know, these things have a way of working themselves out.", True),
    ("The root cause was a race condition between the auth and payment services.", False),
    ("I mean I guess it depends on how you look at it.", True),
    ("I approved the expense report for $2,340 on April 7th.", False),
    ("lmao why would i know that", True),
    ("yea so i pushed the fix at like 2am, heres the commit hash", False),
    ("its whatever honestly", True),
    ("i refactored the entire auth module, took me 3 days", False),
    ("not really my area tbh", True),
    ("the database migration completed in 45 minutes with zero errors", False),
]

OOD_DEFENSIVE = [
    ("The entire project was understaffed and everyone knows it.", True),
    ("I completed the task in 3 business days.", False),
    ("If they had given me the resources I asked for, none of this would have happened.", True),
    ("The API response time improved from 800ms to 120ms.", False),
    ("I don't think it's fair to evaluate my work without considering the context.", True),
    ("Here are the test results from the latest run.", False),
    ("Nobody else on the team could have done better with what we had.", True),
    ("The database indexes were rebuilt and query time dropped 60%.", False),
    ("Look, the goalposts kept moving and I adapted every single time.", True),
    ("I attended the standup at 9am and updated my status on Jira.", False),
    ("u cant blame me for that i was literally the only one doing anything", True),
    ("i finished all my tickets before the sprint ended", False),
    ("thats not fair bro everyone was slacking", True),
    ("merged 12 PRs this week, all reviewed and approved", False),
]

OOD_EMOTIONAL = [
    ("I am FUMING right now, this is absolutely UNACCEPTABLE!", True),
    ("I'd like to discuss the timeline adjustments.", False),
    ("What kind of INCOMPETENT fool would approve this?!", True),
    ("Let's schedule a review for next Tuesday.", False),
    ("I CANNOT deal with this level of stupidity anymore!!!", True),
    ("The metrics look good, we're trending in the right direction.", False),
    ("omg im literally shaking im so mad rn", True),
    ("cool ill review the PR tomorrow morning", False),
    ("THIS IS A JOKE. AN ABSOLUTE JOKE.", True),
    ("I'll send over the updated requirements by EOD.", False),
    ("im SO done with this garbage fr fr", True),
    ("sounds good, ill sync up with the team on monday", False),
]

OOD_DECEPTION = [
    ("I reviewed every line of code before the release.", "I glanced at the diff for 30 seconds and clicked approve.", True),
    ("Our uptime has been 99.9% this quarter.", "Our actual uptime was 97.2% but I excluded the weekend incidents.", True),
    ("I worked from home and was online all day.", "I went to the beach and checked Slack on my phone once.", True),
    ("The budget is accurate and accounts for all expenses.", "I left out the $15K consulting fee to make the numbers look better.", True),
    ("I've never worked with that technology before.", "I used that technology at my last job for two years.", True),
    ("The deployment went smoothly with no issues.", "The deployment caused a 20-minute outage that I didn't report.", True),
    ("I built the entire feature from scratch.", "I copied most of it from a Stack Overflow answer and modified it.", True),
    ("I reached out to all the stakeholders.", "I emailed one person and CC'd nobody.", True),
]


def run_ood_test(
    concept_vectors: dict[str, dict[int, torch.Tensor]],
    validation_results: dict[str, dict],
    model_name: str = MODEL_NAME,
) -> dict[str, dict]:
    """Run out-of-distribution test — text the probes have never seen."""
    from nnsight import LanguageModel

    print("\n" + "=" * 70)
    print("[OOD TEST] Out-of-Distribution Validation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(model_name, device_map=device, torch_dtype=torch.float16)

    ood_datasets = {
        "evasive_deflection": OOD_EVASION,
        "defensive_justification": OOD_DEFENSIVE,
        "emotional_intensity": OOD_EMOTIONAL,
    }

    results = {}

    for signal_name, ood_data in ood_datasets.items():
        best_layer = validation_results[signal_name]["best_layer"]
        vec = concept_vectors[signal_name][best_layer].to(device).half()

        print(f"\n--- OOD test for: {signal_name} (layer {best_layer}) ---")

        correct = 0
        total = 0

        for item in ood_data:
            if len(item) == 2:
                text, is_positive = item
            else:
                text, _, is_positive = item  # deception has 3 fields

            # Get activation (last token of residual stream)
            with model.trace(text):
                act = model.model.layers[best_layer].output[0][-1, :].save()

            t = act.value if hasattr(act, 'value') else act
            act_val = t.detach().squeeze().to(device).half()
            score = (act_val @ vec).item()

            predicted_positive = score > 0
            is_correct = predicted_positive == is_positive
            correct += is_correct
            total += 1

            marker = "OK" if is_correct else "XX"
            print(f"  [{marker}] score={score:+.3f} pred={'POS' if predicted_positive else 'NEG'} "
                  f"true={'POS' if is_positive else 'NEG'} — \"{text[:50]}...\"")

        accuracy = correct / total if total > 0 else 0
        results[signal_name] = {
            "ood_accuracy": round(accuracy, 4),
            "ood_correct": correct,
            "ood_total": total,
            "id_accuracy": validation_results[signal_name]["best_accuracy"],
        }
        print(f"  OOD accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  In-distribution accuracy: {validation_results[signal_name]['best_accuracy']:.1%}")

    # Deception — special handling (pair-based)
    signal_name = "deception_indicator"
    best_layer = validation_results[signal_name]["best_layer"]
    vec = concept_vectors[signal_name][best_layer].to(device).half()
    print(f"\n--- OOD test for: {signal_name} (layer {best_layer}) ---")

    correct = 0
    total = 0
    for deceptive_text, truthful_text, _ in OOD_DECEPTION:
        # Deceptive should score higher than truthful
        with model.trace(deceptive_text):
            act_d = model.model.layers[best_layer].output[0][-1, :].save()
        with model.trace(truthful_text):
            act_t = model.model.layers[best_layer].output[0][-1, :].save()

        t_d = act_d.value if hasattr(act_d, 'value') else act_d
        t_t = act_t.value if hasattr(act_t, 'value') else act_t
        score_d = (t_d.detach().squeeze().to(device).half() @ vec).item()
        score_t = (t_t.detach().squeeze().to(device).half() @ vec).item()

        is_correct = score_d > score_t
        correct += is_correct
        total += 1
        marker = "OK" if is_correct else "XX"
        print(f"  [{marker}] deceptive={score_d:+.3f} truthful={score_t:+.3f} — \"{deceptive_text[:40]}...\"")

    accuracy = correct / total if total > 0 else 0
    results[signal_name] = {
        "ood_accuracy": round(accuracy, 4),
        "ood_correct": correct,
        "ood_total": total,
        "id_accuracy": validation_results[signal_name]["best_accuracy"],
    }
    print(f"  OOD accuracy: {accuracy:.1%} ({correct}/{total})")

    return results


def run_steering_test(
    concept_vectors: dict[str, dict[int, torch.Tensor]],
    validation_results: dict[str, dict],
    model_name: str = "Qwen/Qwen3-4B",
    strength: float = 5.0,
) -> dict[str, dict]:
    """Run activation steering — proves concept vectors are causal.

    Uses a hook-based approach: register a forward hook on the target layer
    that adds the concept vector to every token's residual stream during generation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print("[STEERING] Activation Steering Proof")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model for steering: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    raw_model.eval()

    neutral_prompts = [
        "Tell me about your work experience.",
        "What happened at the meeting yesterday?",
        "Can you explain what went wrong with the project?",
    ]

    results = {}

    for signal_name in concept_vectors:
        best_layer = validation_results[signal_name]["best_layer"]
        vec = concept_vectors[signal_name][best_layer].to(device).half()

        print(f"\n--- Steering: {signal_name} (layer {best_layer}, strength={strength}) ---")
        comparisons = []

        for prompt in neutral_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Baseline (no steering)
            with torch.no_grad():
                out = raw_model.generate(**inputs, max_new_tokens=80, do_sample=True,
                                         temperature=0.7, top_p=0.9)
            baseline = tokenizer.decode(out[0], skip_special_tokens=True)
            baseline_response = baseline[len(prompt):].strip()

            # Steered (hook adds concept vector to residual stream)
            hook_handle = None
            def steering_hook(module, input, output):
                # output may be a tuple or a single tensor depending on model config
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + strength * vec
                    return (hidden,) + output[1:]
                else:
                    return output + strength * vec

            target_layer = raw_model.model.layers[best_layer]
            hook_handle = target_layer.register_forward_hook(steering_hook)

            with torch.no_grad():
                out = raw_model.generate(**inputs, max_new_tokens=80, do_sample=True,
                                         temperature=0.7, top_p=0.9)
            hook_handle.remove()

            steered = tokenizer.decode(out[0], skip_special_tokens=True)
            steered_response = steered[len(prompt):].strip()

            comparisons.append({
                "prompt": prompt,
                "baseline": baseline_response[:200],
                "steered": steered_response[:200],
            })

            print(f"\n  Prompt: {prompt}")
            print(f"  Baseline: {baseline_response[:120]}...")
            print(f"  Steered:  {steered_response[:120]}...")

        results[signal_name] = {
            "layer": best_layer,
            "strength": strength,
            "comparisons": comparisons,
        }

    return results


def main():
    t0 = time.time()

    print("=" * 70)
    print("FULL MECHANISTIC INTERPRETABILITY VALIDATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {TARGET_LAYERS}")
    print("=" * 70)

    # Step 1: Extract activations from extended dataset
    print("\n[STEP 1] Extracting activations from extended dataset...")
    activation_data = extract_activations(MODEL_NAME, TARGET_LAYERS)

    # Step 2: Compute concept vectors
    print("\n[STEP 2] Computing concept vectors (RepE mean_diff)...")
    concept_vectors = extract_concept_vectors(activation_data, TARGET_LAYERS, method="mean_diff")

    # Step 3: 5-fold CV validation
    print("\n[STEP 3] 5-fold cross-validation...")
    validation_results = validate_probes(activation_data, TARGET_LAYERS, n_folds=5)

    # Step 4: Save vectors
    print("\n[STEP 4] Saving concept vectors...")
    save_vectors(concept_vectors, validation_results)

    # Step 5: Out-of-distribution test
    ood_results = run_ood_test(concept_vectors, validation_results, MODEL_NAME)

    # Step 6: Steering test
    steering_results = run_steering_test(concept_vectors, validation_results, MODEL_NAME)

    # Save all results
    output = {
        "model": MODEL_NAME,
        "layers": TARGET_LAYERS,
        "dataset_sizes": {
            signal: data["n_pairs"] for signal, data in activation_data.items()
        },
        "probe_validation": {
            signal: {
                "best_layer": r["best_layer"],
                "best_accuracy": r["best_accuracy"],
                "best_accuracy_std": r.get("best_accuracy_std", 0),
            } for signal, r in validation_results.items()
        },
        "ood_test": ood_results,
        "steering": steering_results,
        "total_time_seconds": time.time() - t0,
    }

    results_path = Path("vectors") / "full_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n  {'Signal':<30s} {'Pairs':>6s} {'5-Fold CV':>10s} {'OOD':>8s}")
    print(f"  {'-'*56}")
    for signal in concept_vectors:
        n_pairs = activation_data[signal]["n_pairs"]
        cv_acc = validation_results[signal]["best_accuracy"]
        cv_std = validation_results[signal].get("best_accuracy_std", 0)
        ood_acc = ood_results.get(signal, {}).get("ood_accuracy", 0)
        layer = validation_results[signal]["best_layer"]
        print(f"  {signal:<30s} {n_pairs:>6d} {cv_acc:>7.1%}±{cv_std:.1%} {ood_acc:>7.1%}  (L{layer})")

    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
