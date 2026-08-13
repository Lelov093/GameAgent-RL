# Final Results

This document is the final evidence summary for GameAgent-RL V1. Frozen final performance and Development sample-efficiency evidence are kept separate. The replication unit for learned policies is the training seed, not the individual evaluation episode.

## Protocol

| Item | Track A | Track B |
|---|---|---|
| Environment | DoorKey-5x5 | MemoryS11 + Research Change 001 |
| Conditions | A0, A0-M, A1, A2 | B0, B1, B2, B2-R |
| Training seeds | 1001–1005 | 1001–1005 |
| Training budget | 100k steps | 500k steps |
| Frozen environment seeds | 3001–3100 | 3001–3100 |
| Learned-policy evaluation | deterministic final checkpoint | deterministic final checkpoint |
| Episodes per learned condition | 500 | 500 |

Frozen evidence did not influence training, tuning, configuration, checkpoint selection, or additional experiments.

## Track A — Constrained Action Policy Learning

### Frozen aggregate

| Condition | Episodes | Success | Mean return | Mean length | Invalid action rate |
|---|---:|---:|---:|---:|---:|
| A0 Random | 500 | 0.254 | 0.1115 | 226.08 | 0.5316 |
| A0-M Masked Random | 500 | 0.550 | 0.2829 | 186.68 | 0.0000 |
| A1 Own PPO | 500 | 0.956 | 0.9219 | 20.47 | 0.5356 |
| A2 Own Masked PPO | 500 | 1.000 | 0.9641 | 9.98 | 0.0000 |

### Frozen per-training-seed robustness

| Seed | A1 success | A2 success | A1 invalid | A2 invalid |
|---|---:|---:|---:|---:|
| 1001 | 1.00 | 1.00 | 0.0000 | 0.0000 |
| 1002 | 0.97 | 1.00 | 0.4438 | 0.0000 |
| 1003 | 0.90 | 1.00 | 0.7395 | 0.0000 |
| 1004 | 1.00 | 1.00 | 0.0000 | 0.0000 |
| 1005 | 0.91 | 1.00 | 0.7168 | 0.0000 |

A2 matched or exceeded A1 success for every paired training seed and produced zero illegal actions in all 500 Frozen episodes. A1 had 22 Frozen failures: 16 timeouts before key pickup and 6 key-picked/door-locked failures.

### Development sample efficiency

These metrics come from the frozen Development learning curves, not the Frozen final panel.

| Condition | Mean normalized AUC | SD across seeds | Steps-to-80 |
|---|---:|---:|---|
| A1 | 0.6475 | 0.0368 | 50k for all 5 seeds |
| A2 | 0.8388 | 0.0568 | 25k for all 5 seeds |

A2 had higher paired AUC in every seed and reached the fixed 0.80 threshold 25k steps earlier in every seed.

### RQ-A closure

- **H-A1 — Supported.** A2 and A0-M had zero illegal actions; A1 and A0 did not. The legality mechanism works independently of policy quality.
- **H-A2 — Supported.** A2 improved Development AUC and Steps-to-80 under the controlled A1/A2 setup.
- **H-A3 — Supported.** A2 improved Frozen success from `0.956` to `1.000`, mean return from `0.9219` to `0.9641`, and mean length from `20.47` to `9.98`.

**RQ-A conclusion:** on DoorKey-5x5, legality-only Action Masking removed invalid decisions and improved both learning efficiency and final task performance without adding planning guidance.

## Track B — Memory under Partial Observability

### Frozen aggregate

| Condition | Episodes | Success | Mean return | Mean length | Decision reach | Memory accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B0 Random | 100 | 0.330 | 0.2087 | 402.21 | 0.650 | 0.508 |
| B1 Feed-forward PPO | 500 | 0.450 | 0.4433 | 10.00 | 1.000 | 0.450 |
| B2 RecurrentPPO | 500 | 0.560 | 0.5506 | 10.93 | 1.000 | 0.560 |
| B2-R state reset | 500 | 0.458 | 0.4505 | 10.98 | 1.000 | 0.458 |

B0 Memory Accuracy is conditional on the 65 episodes that reached the decision. For B1/B2/B2-R, Decision Reach was 1.00, so Success equals Memory Accuracy.

### Frozen per-training-seed result

| Seed | B1 memory | B2 memory | B2-R memory | B2 − B2-R |
|---|---:|---:|---:|---:|
| 1001 | 0.45 | 0.45 | 0.45 | 0.00 |
| 1002 | 0.45 | 1.00 | 0.49 | 0.51 |
| 1003 | 0.45 | 0.45 | 0.45 | 0.00 |
| 1004 | 0.45 | 0.45 | 0.45 | 0.00 |
| 1005 | 0.45 | 0.45 | 0.45 | 0.00 |

Across training seeds, Memory Accuracy was `0.450 ± 0.000` for B1, `0.560 ± 0.220` for B2, and `0.458 ± 0.016` for B2-R.

### Causal evidence and instability

B1 and four B2 seeds always selected the lower branch. Their `0.45` result equals the Frozen panel's lower-branch prior and is not memory competence. B2 seed1002 selected cue-dependent branches correctly in all 100 Frozen episodes. Resetting its hidden and cell state at the contract trigger reduced both Memory Accuracy and Success from `1.00` to `0.49`, converting 51 paired successes into failures.

This is direct causal evidence that seed1002's behavior depended on recurrent state. It is not evidence that RecurrentPPO learned memory reliably: four of five B2 seeds stayed at the branch prior, and the full aggregate improvement over B1 is attributable to seed1002.

### RQ-B closure

- **H-B1 — Partially Supported.** One recurrent seed developed perfect memory-dependent decisions with causal state-reset evidence; four did not.
- **H-B2 — Partially Supported.** Aggregate Frozen success increased from `0.45` to `0.56`, but only one paired seed improved.
- **H-B3 — Not Evaluated.** MemoryS13 was optional and outside the completed base gate.
- **Stable cross-seed memory competence — Not Supported.** Recurrence showed capability but high seed instability under the frozen setup.

**RQ-B conclusion:** RecurrentPPO can learn and causally use the cue on MemoryS11, but it did not do so reliably across seeds; the appropriate result is policy-specific memory evidence plus a negative stable-competence conclusion.

## Evidence map

- Public figures: `assets/figures/`
- Public representative replays: `assets/replays/`
- Track A raw local evaluation: `artifacts/track_a/frozen_evaluation/`
- Track B raw local evaluation: `artifacts/track_b/frozen_evaluation/`
- Formal checkpoints and curves: `artifacts/track_a/formal_training/`, `artifacts/track_b/formal_training/`
- Frozen configurations: `configs/ppo_v1_frozen.yaml`, `configs/reference_track_b.toml`

Local `artifacts/` is intentionally git-ignored because it contains checkpoints and raw run outputs. Internal research contracts, phase records, and development reports are also excluded from the public repository. Figures, representative GIFs, exact aggregate tables, configs, and reproduction code remain public repository material.

## Claim boundary

The evidence supports a DoorKey-5x5 Action Masking result and a seed-specific MemoryS11 recurrent-state result. It does not establish cross-environment generalization, stable recurrent memory learning, superior recurrent performance on most seeds, or claims about architectures not evaluated in V1.
