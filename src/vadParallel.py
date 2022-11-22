import multiprocessing
import threading
import time
from src.vad import AbstractTranscription, TranscriptionConfig
from src.whisperContainer import WhisperCallback

from multiprocessing import Pool

from typing import List
import os


class ParallelContext:
    def __init__(self, num_processes: int = None, auto_cleanup_timeout_seconds: float = None):
        self.num_processes = num_processes
        self.auto_cleanup_timeout_seconds = auto_cleanup_timeout_seconds
        self.lock = threading.Lock()

        self.ref_count = 0
        self.pool = None
        self.cleanup_timer = None

    def get_pool(self):
        # Initialize pool lazily
        if (self.pool is None):
            context = multiprocessing.get_context('spawn')
            self.pool = context.Pool(self.num_processes)

        self.ref_count = self.ref_count + 1

        if (self.auto_cleanup_timeout_seconds is not None):
            self._stop_auto_cleanup()

        return self.pool

    def return_pool(self, pool):
        if (self.pool == pool and self.ref_count > 0):
            self.ref_count = self.ref_count - 1

            if (self.ref_count == 0):
                if (self.auto_cleanup_timeout_seconds is not None):
                    self._start_auto_cleanup()

    def _start_auto_cleanup(self):
        if (self.cleanup_timer is not None):
            self.cleanup_timer.cancel()
        self.cleanup_timer = threading.Timer(self.auto_cleanup_timeout_seconds, self._execute_cleanup)
        self.cleanup_timer.start()

        print("Started auto cleanup of pool in " + str(self.auto_cleanup_timeout_seconds) + " seconds")

    def _stop_auto_cleanup(self):
        if (self.cleanup_timer is not None):
            self.cleanup_timer.cancel()
            self.cleanup_timer = None

            print("Stopped auto cleanup of pool")

    def _execute_cleanup(self):
        print("Executing cleanup of pool")

        if (self.ref_count == 0):
            self.close()

    def close(self):
        self._stop_auto_cleanup()

        if (self.pool is not None):
            print("Closing pool of " + str(self.num_processes) + " processes")
            self.pool.close()
            self.pool.join()
        self.pool = None

class ParallelTranscriptionConfig(TranscriptionConfig):
    def __init__(self, device_id: str, override_timestamps, initial_segment_index, copy: TranscriptionConfig = None):
        super().__init__(copy.non_speech_strategy, copy.segment_padding_left, copy.segment_padding_right, copy.max_silent_period, copy.max_merge_size, copy.max_prompt_window, initial_segment_index)
        self.device_id = device_id
        self.override_timestamps = override_timestamps
    
class ParallelTranscription(AbstractTranscription):
    def __init__(self, sampling_rate: int = 16000):
        super().__init__(sampling_rate=sampling_rate)

    
    def transcribe_parallel(self, transcription: AbstractTranscription, audio: str, whisperCallable: WhisperCallback, config: TranscriptionConfig, devices: List[str], parallel_context: ParallelContext = None):
        # First, get the timestamps for the original audio
        merged = transcription.get_merged_timestamps(audio, config)

        # Split into a list for each device
        # TODO: Split by time instead of by number of chunks
        merged_split = list(self._split(merged, len(devices)))

        # Parameters that will be passed to the transcribe function
        parameters = []
        segment_index = config.initial_segment_index

        for i in range(len(merged_split)):
            device_segment_list = list(merged_split[i])
            device_id = devices[i]

            if (len(device_segment_list) <= 0):
                continue

            print("Device " + device_id + " (index " + str(i) + ") has " + str(len(device_segment_list)) + " segments")

            # Create a new config with the given device ID
            device_config = ParallelTranscriptionConfig(devices[i], device_segment_list, segment_index, config)
            segment_index += len(device_segment_list)

            parameters.append([audio, whisperCallable, device_config]);

        merged = {
            'text': '',
            'segments': [],
            'language': None
        }

        created_context = False

        # Spawn a separate process for each device
        try:
            if (parallel_context is None):
                parallel_context = ParallelContext(len(devices))
                created_context = True

            # Get a pool of processes
            pool = parallel_context.get_pool()

            # Run the transcription in parallel
            results = pool.starmap(self.transcribe, parameters)

            for result in results:
                # Merge the results
                if (result['text'] is not None):
                    merged['text'] += result['text']
                if (result['segments'] is not None):
                    merged['segments'].extend(result['segments'])
                if (result['language'] is not None):
                    merged['language'] = result['language']

        finally:
            # Return the pool to the context
            if (parallel_context is not None):
                parallel_context.return_pool(pool)
            # Always close the context if we created it
            if (created_context):
                parallel_context.close()

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

    def _split(self, a, n):
        """Split a list into n approximately equal parts."""
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

