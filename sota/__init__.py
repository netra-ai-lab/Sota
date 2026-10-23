"""SOTA: compact Khmer and English speech recognition (structurally pruned Omnilingual ASR CTC).

    import sota
    asr = sota.load_model("sota-base")          # or "sota-large", a Hugging Face id, a folder
    print(asr.transcribe("speech.wav"))
"""

from .asr import SOTA, load_model, transcribe
from .hub import available_models

__version__ = "0.1.0"
__all__ = ["SOTA", "load_model", "transcribe", "available_models", "__version__"]
