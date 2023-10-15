from typing import List

import ffmpeg
from src.config import ModelConfig
from src.hooks.progressListener import ProgressListener
from src.modelCache import ModelCache
from src.prompts.abstractPromptStrategy import AbstractPromptStrategy
from src.whisper.abstractWhisperContainer import AbstractWhisperCallback, AbstractWhisperContainer

class DummyWhisperContainer(AbstractWhisperContainer):
    def __init__(self, model_name: str, device: str = None, compute_type: str = "float16",
                       download_root: str = None,
                       cache: ModelCache = None, models: List[ModelConfig] = []):
        super().__init__(model_name, device, compute_type, download_root, cache, models)

    def ensure_downloaded(self):
        """
        Ensure that the model is downloaded. This is useful if you want to ensure that the model is downloaded before
        passing the container to a subprocess.
        """
        print("[Dummy] Ensuring that the model is downloaded")

    def _create_model(self):
        print("[Dummy] Creating dummy whisper model " + self.model_name + " for device " + str(self.device))
        return None

    def create_callback(self, language: str = None, task: str = None, 
                        prompt_strategy: AbstractPromptStrategy = None, 
                        **decodeOptions: dict) -> AbstractWhisperCallback:
        """
        Create a WhisperCallback object that can be used to transcript audio files.

        Parameters
        ----------
        language: str
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        prompt_strategy: AbstractPromptStrategy
            The prompt strategy to use. If not specified, the prompt from Whisper will be used.
        decodeOptions: dict
            Additional options to pass to the decoder. Must be pickleable.

        Returns
        -------
        A WhisperCallback object.
        """
        return DummyWhisperCallback(self, language=language, task=task, prompt_strategy=prompt_strategy, **decodeOptions)
    
class DummyWhisperCallback(AbstractWhisperCallback):
    def __init__(self, model_container: DummyWhisperContainer, **decodeOptions: dict):
        self.model_container = model_container
        self.decodeOptions = decodeOptions
        
    def invoke(self, audio, segment_index: int, prompt: str, detected_language: str, progress_listener: ProgressListener = None):
        """
        Peform the transcription of the given audio file or data.

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor]
            The audio file to transcribe, or the audio data as a numpy array or torch tensor.
        segment_index: int
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        progress_listener: ProgressListener
            A callback to receive progress updates.
        """
        print("[Dummy] Invoking dummy whisper callback for segment " + str(segment_index))

        # Estimate length
        if isinstance(audio, str):
            audio_length = ffmpeg.probe(audio)["format"]["duration"]
        # Format is pcm_s16le at a sample rate of 16000, loaded as a float32 array.
        else:
            audio_length = len(audio) / 16000

        # Convert the segments to a format that is easier to serialize
        whisper_segments = [{
            "text": "Dummy text for segment " + str(segment_index),
            "start": 0,
            "end": audio_length,

            # Extra fields added by faster-whisper
            "words": []
        }]

        result = {
            "segments": whisper_segments,
            "text": "Dummy text for segment " + str(segment_index),
            "language": "en" if detected_language is None else detected_language,

            # Extra fields added by faster-whisper
            "language_probability": 1.0,
            "duration": audio_length,
        }

        if progress_listener is not None:
            progress_listener.on_finished()
        return result