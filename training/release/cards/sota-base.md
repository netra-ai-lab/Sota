---
license: apache-2.0
language:
- km
- en
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
- omnilingual-asr
- ctc
- wav2vec2
- structured-pruning
- khmer
---

# sota-base

A **73M-parameter** Khmer + English CTC speech recogniser, made by structured pruning of
Meta's Omnilingual ASR `omniASR_CTC_300M` to **80% encoder sparsity**. It is the Base size of a pair,
named after DiariZen's Base-s80 / Large-s80 pruned WavLM models. The other size is
[Darayut/sota-large](https://huggingface.co/Darayut/sota-large).

**Code:** [github.com/netra-ai-lab/Sota](https://github.com/netra-ai-lab/Sota), with the `sota` inference package and the full training recipe.

## Results

Test sets; greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | Params | GMAC / s audio | DDD (km) | FLEURS-km | FLEURS-en | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|---|---|
| `omniASR_CTC_300M` (stock) | 325M | 18.3 | 9.65 | 16.26 | – | 6.53 | 14.45 |
| [omniASR-CTC-300M-km-en](https://huggingface.co/Darayut/omniASR-CTC-300M-km-en) (fine-tuned teacher) | 325M | 18.3 | 3.68 | 12.64 | 19.51 | 5.89 | 13.86 |
| **sota-base** | **73M** | **5.8** | **2.60** | **16.39** | **30.24** | **11.15** | **27.36** |

- **Khmer:** on DDD it is more accurate than both its unpruned fine-tuned teacher and the stock model.
- **English:** it keeps much less of its teacher's accuracy, with about 1.5–2× the teacher's WER.
- **In-domain test sets:** FLEURS-km and FLEURS-en train are in the training mix, so their test sets are in-domain.

<p align="center"><img src="https://huggingface.co/Darayut/sota-base/resolve/main/assets/tradeoff.png" width="100%" alt="Error rate against speed-up for each fine-tuned teacher and its 80%-sparse model"></p>

*Speed against error.* Each arrow runs from a fine-tuned teacher to its pruned model (solid: large, dashed: base). Marker area is the parameter count, and each label gives the error and its change against the teacher. Speed is relative to the fine-tuned 1B (real-time factor).

## Architecture

<p align="center"><img src="https://huggingface.co/Darayut/sota-base/resolve/main/assets/pruning.png" width="100%" alt="The encoder before and after pruning: CNN kept, attention heads and FFN units removed"></p>

*What is pruned*, shown for sota-large: whole attention heads (MHA) and FFN units are removed from every transformer layer, and the CNN front end is kept. Each cell is a quarter of a layer's heads or units.

- **Base:** a wav2vec2-style CTC encoder with Omnilingual's 9,812-entry character vocabulary, which covers Khmer and Latin script.
- **Pruned parts:** per-layer attention heads and FFN units, so every layer has its own width. `config.json` lists the exact `num_heads` and `ffn_dims`; the encoder has 63M parameters.
- **Unchanged parts:** the CNN feature extractor and the output layer.
- **Loading:** the layer widths are non-uniform, so fairseq2 / `omnilingual-asr` cannot load it. Use the `sota` package (see Usage) or the code bundled in this repo.

## How it was made

The recipe follows *Fine-tune Before Structured Pruning* (arXiv 2505.24111), with the pruning method of
DPHuBERT (arXiv 2305.17651) as used in DiariZen.

1. **Fine-tune.** Train `omniASR_CTC_300M` with CTC on the training mix. The result is the teacher, [omniASR-CTC-300M-km-en](https://huggingface.co/Darayut/omniASR-CTC-300M-km-en).
2. **Prune** (12k steps). Hard Concrete L0 gates sit on the attention heads and FFN units. An augmented-Lagrangian constraint ramps the target sparsity to 80%. The pruned model is distilled to the teacher with an L1 − cosine loss on hidden states at layers 6, 12, 18 and 24.
3. **Cut.** Remove the pruned units for good.
4. **Distil** (5k steps). Keep the mask frozen and continue the layer distillation.
5. **Recover** (16k steps). Train with CTC plus a frame-level KL divergence to the teacher's output distribution (temperature 1, weight 1).

**Training mix**, 50/50 by language:
- Khmer: DDD train and FLEURS-km train.
- English: LibriSpeech train-clean-100 + 360 and FLEURS-en train.
- FLEURS is fixed at 20% of each language's audio.

## Usage

```bash
pip install git+https://github.com/netra-ai-lab/SOTA
```

```python
import sota

asr = sota.load_model("sota-base")         # downloads this repo's weights once
print(asr.transcribe("audio.wav"))
print(asr.transcribe(["a.wav", "b.flac"]))
```

```bash
sota transcribe audio.wav --model sota-base
```

Without installing the package, the code bundled in this repo also works:

```bash
huggingface-cli download Darayut/sota-base --local-dir sota-base
pip install torch torchaudio soundfile safetensors
python sota-base/inference.py audio.wav
```

Audio of any sample rate is accepted; it is mixed to mono and resampled to 16 kHz. Output text is in the
Omnilingual character set: lowercase, with no Khmer punctuation.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | weights |
| `config.json` | architecture, including per-layer `num_heads` and `ffn_dims` |
| `vocab.json` | the Omnilingual character vocabulary, in id order |
| `omniprune/` | model and text code (`model.py`, `hardconcrete.py`, `text.py`) |
| `inference.py` | load and transcribe |

The released files reproduce the training checkpoint's logits exactly: the maximum difference is 0.

## License and attribution

- **Weights:** derived from Meta's Omnilingual ASR `omniASR_CTC_300M` (Apache 2.0) and released under Apache 2.0.
- **`omniprune/hardconcrete.py`:** ported from DiariZen (MIT), itself derived from DPHuBERT and FLOP.
