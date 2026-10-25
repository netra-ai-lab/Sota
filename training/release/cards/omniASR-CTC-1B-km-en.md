---
license: apache-2.0
language:
- km
- en
library_name: fairseq2
pipeline_tag: automatic-speech-recognition
tags:
- omnilingual-asr
- ctc
- wav2vec2
- khmer
---

# omniASR-CTC-1B-km-en

[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) `omniASR_CTC_1B` (975M parameters, 48 layers, dim 1280, 20 heads, FFN 5120) fine-tuned with CTC for
**Khmer and English** speech recognition.

This is the Stage 1 model of a structured-pruning study (fine-tune first, then prune to 80%
sparsity with distillation, following *Fine-tune Before Structured Pruning*, arXiv 2505.24111).
It serves as the teacher and starting point for the pruned models.

**Code:** [github.com/netra-ai-lab/Sota](https://github.com/netra-ai-lab/Sota).

## Results

Greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | DDD (km) | FLEURS-km test | FLEURS-en test | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|
| `omniASR_CTC_1B` (stock) | 5.95 | 11.26 | 12.38 | 4.40 | 8.87 |
| **this model** | **2.47** | **9.44** | **11.94** | **3.61** | **8.21** |

Dev sets: DDD dev 3.40 CER, FLEURS-km dev 8.83 CER, FLEURS-en dev 13.68 WER, LibriSpeech dev-clean 3.72 WER.

Text is normalised to the Omnilingual character vocabulary (lowercase, no punctuation).
FLEURS-km and FLEURS-en train splits are in the training mix, so their test sets are in-domain.

## Training

- **Data**, mixed 50/50 by language:
  - Khmer: DDD train and FLEURS-km train
  - English: LibriSpeech train-clean-100 + train-clean-360 and FLEURS-en train
  - FLEURS is fixed at 20% of each language's audio
- **Objective:** CTC
- **Schedule:** 6,000 steps, batches of up to 160 s of audio
- **Learning rates:** 1e-5 for the encoder, 1e-4 for the output projection; 500 warm-up steps, then decay to 5%
- **Regularisation:** CNN feature extractor frozen; layerdrop 0.05; FFN dropout 0.1; weight decay 0.01

## Files

| File | Format |
|---|---|
| `omniASR-CTC-1B-km-en.pt` | fairseq2 checkpoint; drop-in replacement for the original `omniASR_CTC_1B` checkpoint |
| `omniASR_tokenizer.model` | the unchanged Omnilingual character tokenizer |
| `asset_card.yaml` | fairseq2 asset card for the two files above |
| `model.safetensors` + `config.json` | the same weights in the `omniprune` layout (fairseq2 parameter names, position-conv weight norm as a parametrization); `config.json` is its `ArchConfig` |

The fairseq2 checkpoint was verified by reloading it with fairseq2 and comparing logits
against the training model: the maximum difference was exactly zero.

## Usage (sota)

```bash
pip install git+https://github.com/netra-ai-lab/SOTA
```

```python
import sota

asr = sota.load_model("teacher-1b")      # downloads this repo's weights once
print(asr.transcribe("audio.wav"))
```

## Usage (omnilingual-asr)

```bash
pip install omnilingual-asr
huggingface-cli download Darayut/omniASR-CTC-1B-km-en --local-dir omniASR-CTC-1B-km-en
cd omniASR-CTC-1B-km-en && sed -i "s#/path/to/omniASR-CTC-1B-km-en#$PWD#" asset_card.yaml
export FAIRSEQ2_USER_ASSET_DIR=$PWD
```

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipe = ASRInferencePipeline(model_card="omniASR_CTC_1B_km_en")
print(pipe.transcribe(["audio.wav"], batch_size=1))
```

## License and attribution

Fine-tuned from Meta's Omnilingual ASR `omniASR_CTC_1B`, released under Apache 2.0. This model is released
under the same license.
