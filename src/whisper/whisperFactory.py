from typing import List
from src import modelCache
from src.config import ModelConfig
from src.whisper.abstractWhisperContainer import AbstractWhisperContainer

def create_whisper_container(whisper_implementation: str, 
                             model_name: str, device: str = None, download_root: str = None,
                             cache: modelCache = None, models: List[ModelConfig] = []) -> AbstractWhisperContainer:
    print("Creating whisper container for " + whisper_implementation)

    if (whisper_implementation == "whisper"):
        from src.whisper.whisperContainer import WhisperContainer
        return WhisperContainer(model_name, device, download_root, cache, models)
    elif (whisper_implementation == "faster-whisper" or whisper_implementation == "faster_whisper"):
        from src.whisper.fasterWhisperContainer import FasterWhisperContainer
        return FasterWhisperContainer(model_name, device, download_root, cache, models)
    else:
        raise ValueError("Unknown Whisper implementation: " + whisper_implementation)