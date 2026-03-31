# Mechanistic Behavioral Signal Extraction

Extracting deterministic behavioral signals from LLM internal representations using Representation Engineering (RepE) and linear probing. This work validates that deception-related concepts exist as real geometric directions inside transformer neural networks.

## Thesis

LLMs internally represent behavioral concepts like evasion, defensiveness, emotional intensity, and deception as linear directions in their residual stream. We extract these directions as static concept vectors and validate them through probing, out-of-distribution testing, activation steering, and downstream benchmark evaluation. The goal is not detection accuracy. It is proof that the concepts measured by the Kakerou deception engine correspond to real computational structures inside the model, not surface-level word patterns.

## Theoretical Foundation

This work builds on two papers.

Zou et al. (2023), "Representation Engineering," demonstrated that high-level cognitive phenomena exist as population-level representations in transformer residual streams. These representations are extractable via linear probing and PCA on contrast pairs, and can be used both for reading model state and for steering model behavior through activation injection.

Burns et al. (2022), "Eliciting Latent Knowledge," showed that a model's internal beliefs can be separated from its text output. Contrast-Consistent Search finds truth directions in activation space without requiring labeled data. The model's geometry encodes knowledge independently of what it generates.

## Results

We validated on Qwen3-4B using 261 contrast pairs and 5-fold stratified cross-validation.

| Signal | Pairs | 5-Fold CV | OOD Accuracy | Best Layer |
|--------|-------|-----------|--------------|------------|
| Evasion | 77 | 98.7% ±1.6% | 100% (16/16) | 18 |
| Defense | 61 | 92.7% ±7.8% | 64.3% (9/14) | 19 |
| Emotion | 56 | 97.3% ±5.5% | 100% (12/12) | 18 |
| Deception | 67 | 91.8% ±1.5% | 100% (8/8) | 16 |

Evasion, emotion, and deception are genuinely distinct directions with pairwise cosine similarity below 0.3. Defense shows 0.53 overlap with evasion and 0.49 with emotion, which explains its OOD degradation. We address this finding in the Interpretation section below.

Injecting concept vectors during generation produces directionally correct behavioral shifts. Evasive text becomes vaguer and less specific. Emotional text intensifies. This confirms causal influence, not just statistical correlation between the direction and the labeled behavior.

We fed concept vectors into the Kakerou deception detection engine and measured 0.729 AUC on 500 CaSiNo negotiation conversations with 5-fold CV. The hand-tuned regex pipeline achieves 0.739 AUC on the same data. The 95% bootstrap confidence intervals overlap, meaning there is no statistically significant difference between the two approaches.

In isolation, concept vectors alone (no speech acts, no LIWC, no auxiliary analyzers) produce 0.711 classifier AUC. The trajectory features extracted from concept vector probabilities carry real discriminative signal even without the rest of the pipeline.

## Process

### Research Direction

The Kakerou engine measures five behavioral signals to detect deception. These signals were designed from psycholinguistic literature (Pennebaker et al. 2015, Newman et al. 2003) and implemented as regex classifiers and word-rate counters. The system achieves 0.729 AUC on CaSiNo negotiation data, but a core question remained open. Are these signal categories arbitrary heuristics, or do they correspond to real computational structures inside language models?

If the model internally represents "evasion" as a geometric direction in its activation space, that validates the signal taxonomy. The engine would be measuring something computationally real. If the directions don't exist or don't separate cleanly, the engine's accuracy would be a coincidence of surface-level word statistics, and the theoretical foundation would collapse.

### Experimental Judgment

Before trusting any claim about concept vectors, five conditions had to hold.

First, probe accuracy needed real variance. The initial 20-pair experiment produced three perfect 100% results. That was suspicious, not impressive. Accuracy had to come with standard deviations from cross-validation across multiple folds, not a single lucky train/test split on six examples.

Second, the probes had to generalize out-of-distribution. If they only work on text resembling the training data, they learned surface features like word length and punctuation density, not the actual behavioral concept. I required OOD testing on genuinely different text styles, registers, and topics.

Third, the concept vectors had to be orthogonal. If all four vectors point in roughly the same direction, they are just negative sentiment relabeled four times. Pairwise cosine similarity between vectors had to stay below 0.3 for the core signals.

Fourth, I needed causal influence, not just correlation. A probe that classifies correctly shows the information exists in the residual stream. It does not show the model uses that information. Activation steering, where injecting the vector during generation changes the output behavior, was required to demonstrate causation.

Fifth, the vectors had to produce non-trivial AUC when fed into the actual detection benchmark. Scoring well on isolated probing tasks is interesting but insufficient. The vectors needed to function as a working sensor in the real pipeline.

### Implementation

The extraction pipeline, validation suite, dataset construction, Kakerou integration, and visualization code were built with AI assistance (Claude). I designed the experiments, set the validation criteria, interpreted every result, and made all decisions about what to test next, what to trust, and what to discard. The code is the implementation of my experimental design, not the other way around.

### Interpretation and Iteration

Three results changed the direction of the work.

The deception probe jumped from 66.7% to 91.8%. The initial 20-pair dataset only included simple factual lies, things like "I've never been to that restaurant" paired against "I went there last Tuesday." At 66.7% on six test examples, deception looked like it might not be linearly representable at all. I expanded to 67 pairs covering omission-based deception, social deception, strategic negotiation bluffs, and subtle vagueness-versus-specificity contrasts. The model does encode deception. We were asking the wrong questions with too narrow a dataset to find the direction.

Defense failed OOD at 64.3% despite 92.7% in-distribution accuracy. This was the most informative negative result in the entire experiment. Cosine similarity analysis revealed the defense vector has 0.53 overlap with evasion and 0.49 with emotion. It is not an independent concept but a blend of two other signals. Defensiveness depends on conversational context. The same sentence reads as defensive or purely factual depending on what question preceded it. That makes it unsuitable for single-utterance probing. I kept this result rather than discarding it because it validates the methodology. The OOD test and orthogonality check caught exactly the kind of concept that should not generalize from isolated text. A test that passes everything is not a test.

The isolation AUC came back at 0.522 Bayesian but 0.711 classifier. Concept vectors alone produce nearly random Bayesian probability estimates, which initially looked like total failure. But the classifier still extracts 0.711 AUC from the same probability trajectory. The 41-feature extractor learns from the shape of the probability curve across turns, not the absolute values at any single turn. The concept vectors produce meaningfully different trajectories for deceptive versus truthful conversations even when the per-turn probabilities are poorly calibrated. The signal lives in the dynamics, not the magnitude. That finding reframed what the vectors actually contribute. They don't replace the engine's signal extraction. They confirm that the concepts the engine measures exist as real structures in the model's geometry.

## Project Structure

```
mech_interp/
├── dataset.py               # Original 80 contrast pairs (20 per signal)
├── dataset_extended.py       # Extended 261 pairs with diverse registers
├── extract_vectors.py        # Activations, concept vectors, 5-fold CV probes
├── run_full_validation.py    # Full validation with OOD + steering
├── eval_integration.py       # CaSiNo AUC comparison against regex pipeline
├── kakerou_integration.py    # Drop-in MechanisticSensor for Kakerou engine
├── create_story_figures.py   # Visualization generation
├── vectors/                  # Saved concept vectors (.pt) + results (.json)
└── figures/                  # Generated visualizations
```

## Requirements

```
pip install nnsight torch scikit-learn transformers datasets matplotlib
```

GPU with roughly 8GB VRAM required. Tested on an NVIDIA RTX 3080 (10GB).

## Usage

```bash
# Full validation pipeline (extraction + probing + OOD + steering)
python run_full_validation.py

# Downstream benchmark comparison against Kakerou regex pipeline
python eval_integration.py

# Generate figures
python create_story_figures.py
```

The eval_integration script expects the Kakerou engine repo as a sibling directory. All other scripts are self-contained.

## License

AGPLv3
