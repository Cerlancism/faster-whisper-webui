import multiprocessing
from src.vad import AbstractTranscription, TranscriptionConfig
from src.whisperContainer import WhisperCallback

from multiprocessing import Pool

from typing import List
import os

class ParallelTranscriptionConfig(TranscriptionConfig):
    def __init__(self, device_id: str, override_timestamps, initial_segment_index, copy: TranscriptionConfig = None):
        super().__init__(copy.non_speech_strategy, copy.segment_padding_left, copy.segment_padding_right, copy.max_silent_period, copy.max_merge_size, copy.max_prompt_window, initial_segment_index)
        self.device_id = device_id
        self.override_timestamps = override_timestamps
    
class ParallelTranscription(AbstractTranscription):
    def __init__(self, sampling_rate: int = 16000):
        super().__init__(sampling_rate=sampling_rate)

    
    def transcribe_parallel(self, transcription: AbstractTranscription, audio: str, whisperCallable: WhisperCallback, config: TranscriptionConfig, devices: List[str]):
        # First, get the timestamps for the original audio
        merged = transcription.get_merged_timestamps(audio, config)

        # Split into a list for each device
        # TODO: Split by time instead of by number of chunks
        merged_split = self._chunks(merged, len(merged) // len(devices))

        # Parameters that will be passed to the transcribe function
        parameters = []
        segment_index = config.initial_segment_index

        for i in range(len(devices)):
            device_segment_list = merged_split[i]

            # Create a new config with the given device ID
            device_config = ParallelTranscriptionConfig(devices[i], device_segment_list, segment_index, config)
            segment_index += len(device_segment_list)

            parameters.append([audio, whisperCallable, device_config]);

        merged = {
            'text': '',
            'segments': [],
            'language': None
        }

        # Spawn a separate process for each device
        context = multiprocessing.get_context('spawn')

        with context.Pool(len(devices)) as p:
            # Run the transcription in parallel
            results = p.starmap(self.transcribe, parameters)

            for result in results:
                # Merge the results
                if (result['text'] is not None):
                    merged['text'] += result['text']
                if (result['segments'] is not None):
                    merged['segments'].extend(result['segments'])
                if (result['language'] is not None):
                    merged['language'] = result['language']

        return merged

    def get_transcribe_timestamps(self, audio: str, config: ParallelTranscriptionConfig):
        return []

    def get_merged_timestamps(self, audio: str, config: ParallelTranscriptionConfig):
        # Override timestamps that will be processed
        if (config.override_timestamps is not None):
            print("Using override timestamps of size " + str(len(config.override_timestamps)))
            return config.override_timestamps
        return super().get_merged_timestamps(audio, config)

    def transcribe(self, audio: str, whisperCallable: WhisperCallback, config: ParallelTranscriptionConfig):
        # Override device ID
        if (config.device_id is not None):
            print("Using device " + config.device_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id
        return super().transcribe(audio, whisperCallable, config)

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        return [lst[i:i + n] for i in range(0, len(lst), n)]

