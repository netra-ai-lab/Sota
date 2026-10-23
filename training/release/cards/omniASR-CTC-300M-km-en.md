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

# omniASR-CTC-300M-km-en

[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) `omniASR_CTC_300M` (325M parameters, 24 layers, dim 1024, 16 heads, FFN 4096) fine-tuned with CTC for
**Khmer and English** speech recognition.

This is the Stage 1 model of a structured-pruning study (fine-tune first, then prune to 80%
sparsity with distillation, following *Fine-tune Before Structured Pruning*, arXiv 2505.24111).
It serves as the teacher and starting point for the pruned models.

## Results

Greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | DDD-Test (km) | FLEURS-km test | FLEURS-en test | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|
| `omniASR_CTC_300M` (stock) | 9.65 | 16.26 | – | 6.53 | 14.45 |
| **this model** | **3.68** | **12.64** | **19.51** | **5.89** | **13.86** |

Dev sets: DDD dev 4.64 CER, FLEURS-km dev 12.26 CER, FLEURS-en dev 20.78 WER, LibriSpeech dev-clean 5.69 WER.

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
| `omniASR-CTC-300M-km-en.pt` | fairseq2 checkpoint; drop-in replacement for the original `omniASR_CTC_300M` checkpoint |
| `omniASR_tokenizer.model` | the unchanged Omnilingual character tokenizer |
| `asset_card.yaml` | fairseq2 asset card for the two files above |
| `model.safetensors` + `config.json` | the same weights in the `omniprune` layout (fairseq2 parameter names, position-conv weight norm as a parametrization); `config.json` is its `ArchConfig` |

The fairseq2 checkpoint was verified by reloading it with fairseq2 and comparing logits
against the training model: the maximum difference was exactly zero.

## Usage (omnilingual-asr)

```bash
pip install omnilingual-asr
huggingface-cli download Darayut/omniASR-CTC-300M-km-en --local-dir omniASR-CTC-300M-km-en
cd omniASR-CTC-300M-km-en && sed -i "s#/path/to/omniASR-CTC-300M-km-en#$PWD#" asset_card.yaml
export FAIRSEQ2_USER_ASSET_DIR=$PWD
```

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipe = ASRInferencePipeline(model_card="omniASR_CTC_300M_km_en")
print(pipe.transcribe(["audio.wav"], batch_size=1))
```

## License and attribution

Fine-tuned from Meta's Omnilingual ASR `omniASR_CTC_300M`, released under Apache 2.0. This model is released
under the same license.
