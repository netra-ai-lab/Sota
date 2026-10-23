# Results

Greedy CTC decoding with no language model. CER is used for Khmer and WER for English, all in %, and lower is better. Compute is GMAC per second of audio. Every number comes from `training/runs/eval/*.json` and `training/runs/*/*/metrics.jsonl`.

## 1. Test sets

| Model | Params | GMAC/s | DDD-Test | FLEURS-km | FLEURS-en | test-clean | test-other |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| omniASR CTC-1B (stock) | 975M | 50.3 | 5.95 | 11.26 | 12.38 | 4.40 | 8.87 |
| Fine-tuned 1B (teacher) | 975M | 50.3 | 2.47 | 9.44 | 11.94 | 3.61 | 8.21 |
| **sota-large** | **204M** | **12.3** | **1.77** | **11.40** | **19.25** | **6.07** | **16.73** |
| omniASR CTC-300M (stock) | 325M | 18.3 | 9.65 | 16.26 | – | 6.53 | 14.45 |
| Fine-tuned 300M (teacher) | 325M | 18.3 | 3.68 | 12.64 | 19.51 | 5.89 | 13.86 |
| **sota-base** | **73M** | **5.8** | **2.60** | **16.39** | **30.24** | **11.15** | **27.36** |

**Test sets:**

| Set | Utterances | Notes |
| :--- | ---: | :--- |
| DDD-Test | 1,993 | held out from DDD training |
| FLEURS-km test | 771 | |
| FLEURS-en test | 647 | |
| LibriSpeech test-clean | 2,620 | |
| LibriSpeech test-other | 2,939 | |

FLEURS train is in the training mix, so the two FLEURS test sets are in-domain. The stock 300M was not scored on FLEURS-en.

## 2. Ablations on the 1B

| Run | What changes | DDD-Test | FLEURS-km | FLEURS-en | test-clean | test-other |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| A | prune the stock 1B, no Stage 1, no FLEURS | 2.20 | 16.36 | 28.07 | 6.56 | 18.26 |
| B1 | fine-tune first, Stage 3 CTC only (8k) | 2.14 | 12.51 | 21.16 | 7.07 | 18.51 |
| B2 | + logit KD in Stage 3 (8k) | 2.09 | 11.90 | 19.62 | 6.47 | 17.42 |
| **B3** (sota-large) | + Stage 3 for 16k steps | **1.77** | **11.40** | **19.25** | **6.07** | **16.73** |

A → B changes two things at once: the teacher (stock vs fine-tuned) and the data (FLEURS added). Their effects are not separated.

## 3. Stage 2: the pruning run

Dev sets:
- DDD dev and LibriSpeech dev-clean, the first 400 utterances each;
- FLEURS-km dev, 326 utterances;
- FLEURS-en dev, 394 utterances.

| Step | Large sparsity | Large DDD | Large Libri | Base sparsity | Base DDD | Base Libri |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 51.8% | 7.87 | 12.37 | 51.8% | 9.80 | 18.38 |
| 4,000 | 83.1% | 11.95 | 30.32 | 83.0% | 20.03 | 44.97 |
| 6,000 | 73.2% | 7.53 | 14.18 | 76.9% | 13.42 | 27.42 |
| 8,000 | 80.7% | 7.81 | 16.27 | 80.6% | 12.14 | 24.38 |
| 10,000 | 80.3% | 6.85 | 12.58 | 80.1% | 11.43 | 22.23 |
| 12,000 | 80.2% | 7.06 | 12.97 | 80.1% | 9.98 | 20.01 |

- **Constraint behaviour.** The Lagrangian overshoots the target around step 4,000, then settles at 80% from about step 7,000.
- **Experiment A** (stock 1B as init and teacher) ended Stage 2 at 10.71% DDD dev CER, against 7.06% when pruning from the fine-tuned model.

## 4. Dev error through the chain

| Point | Large DDD | Large Libri | Large FLEURS-km | Large FLEURS-en | Base DDD | Base Libri | Base FLEURS-km | Base FLEURS-en |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 7.94 | 4.35 | 10.69 | 14.47 | 10.98 | 6.96 | 15.61 | 21.47 |
| After Stage 1 (teacher) | 3.40 | 3.95 | 8.83 | 13.68 | 4.64 | 5.98 | 12.33 | 20.86 |
| After Stage 2 (pruned) | 7.06 | 12.97 | 18.29 | 33.96 | 9.98 | 20.01 | 25.99 | 44.01 |
| After Stage 2b | 5.60 | 8.46 | 14.64 | 25.53 | 8.61 | 15.49 | 23.42 | 37.91 |
| After Stage 3 (released) | 2.40 | 6.60 | 10.71 | 21.15 | 3.47 | 11.73 | 15.88 | 32.15 |

- **Recovery.** Stages 2b and 3 recover most of what pruning cost, and on DDD both models end below their teachers.
- **English.** It recovers less: the released models' English error is roughly 1.5–2× their teachers'.
- **Training length.** Dev error was still falling at the end of the base model's Stage 3.

## 5. Training data

| Source | Language | Train size | Share of training audio |
| :--- | :--- | :--- | :--- |
| DDD | Khmer | 1,058 h, 4,423 parquet shards | 80% of Khmer |
| FLEURS km_kh train | Khmer | 7.1 h, 1,675 clips | 20% of Khmer (repeated) |
| LibriSpeech train-clean-100 + 360 | English | 464 h | 80% of English |
| FLEURS en_us train | English | 7.5 h, 2,602 clips | 20% of English (repeated) |

The two languages are mixed 50/50 by audio seconds.
