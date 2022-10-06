from abc import ABC, abstractmethod
from collections import Counter
from dis import dis
import re
from typing import Any, Iterator, List, Dict

from pprint import pprint

# Workaround for https://github.com/tensorflow/tensorflow/issues/48797
try:
    import tensorflow as tf
except ModuleNotFoundError:
    # Error handling
    pass

import torch

import ffmpeg
import numpy as np

from src.utils import format_timestamp

# Defaults for Silero
# TODO: Make these configurable?

SPEECH_TRESHOLD = 0.3
MAX_SILENT_PERIOD = 10 # seconds
MAX_MERGE_SIZE = 150 # Do not create segments larger than 2.5 minutes

SEGMENT_PADDING_LEFT = 1 # Start detected text segment early
SEGMENT_PADDING_RIGHT = 1 # End detected segments late

# Whether to attempt to transcribe non-speech
TRANSCRIBE_NON_SPEECH = False

# Minimum size of segments to process
MIN_SEGMENT_DURATION = 1

class AbstractTranscription(ABC):
    def __init__(self, segment_padding_left: int = None, segment_padding_right = None, max_silent_period: int = None, max_merge_size: int = None, transcribe_non_speech: bool = False):
        self.sampling_rate = 16000
        self.segment_padding_left = segment_padding_left
        self.segment_padding_right = segment_padding_right
        self.max_silent_period = max_silent_period
        self.max_merge_size = max_merge_size
        self.transcribe_non_speech = transcribe_non_speech

    def get_audio_segment(self, str, start_time: str = None, duration: str = None):
        return load_audio(str, self.sampling_rate, start_time, duration)

    @abstractmethod
    def get_transcribe_timestamps(self, audio: str):
        """
        Get the start and end timestamps of the sections that should be transcribed by this VAD method.

        Parameters
        ----------
        audio: str
            The audio file.

        Returns
        -------
        A list of start and end timestamps, in fractional seconds.
        """
        return 

    def transcribe(self, audio: str, whisperCallable):
        """
        Transcribe the given audo file.

        Parameters
        ----------
        audio: str
            The audio file.

        whisperCallable: Callable[[Union[str, np.ndarray, torch.Tensor]], dict[str, Union[dict, Any]]]
            The callback that is used to invoke Whisper on an audio file/buffer.

        Returns
        -------
        A list of start and end timestamps, in fractional seconds.
        """

        # get speech timestamps from full audio file
        seconds_timestamps = self.get_transcribe_timestamps(audio)

        padded = self.pad_timestamps(seconds_timestamps, self.segment_padding_left, self.segment_padding_right)
        merged = self.merge_timestamps(padded, self.max_silent_period, self.max_merge_size)

        print("Timestamps:")
        pprint(merged)

        if self.transcribe_non_speech:
            max_audio_duration = float(ffmpeg.probe(audio)["format"]["duration"])

            # Expand segments to include the gaps between them
            merged = self.expand_gaps(merged, total_duration=max_audio_duration)

            print("Transcribing non-speech:")
            pprint(merged)

        result = {
            'text': "",
            'segments': [],
            'language': ""
        }
        languageCounter = Counter()

        # For each time segment, run whisper
        for segment in merged:
            segment_start = segment['start']
            segment_end = segment['end']
            segment_expand_amount = segment.get('expand_amount', 0)

            segment_duration = segment_end - segment_start

            if segment_duration < MIN_SEGMENT_DURATION:
                continue;

            segment_audio = self.get_audio_segment(audio, start_time = str(segment_start), duration = str(segment_duration))

            print("Running whisper from ", format_timestamp(segment_start), " to ", format_timestamp(segment_end), ", duration: ", segment_duration, "expanded: ", segment_expand_amount)
            segment_result = whisperCallable(segment_audio)

            adjusted_segments = self.adjust_whisper_timestamp(segment_result["segments"], adjust_seconds=segment_start, max_source_time=segment_duration)

            # Append to output
            result['text'] += segment_result['text']
            result['segments'].extend(adjusted_segments)

            # Increment detected language
            languageCounter[segment_result['language']] += 1

        if len(languageCounter) > 0:
            result['language'] = languageCounter.most_common(1)[0][0]

        return result
            
    def include_gaps(self, segments: Iterator[dict], min_gap_length: float, total_duration: float):
        result = []
        last_end_time = 0

        for segment in segments:
            segment_start = float(segment['start'])
            segment_end = float(segment['end'])

            if (last_end_time != segment_start):
                delta = segment_start - last_end_time

                if (min_gap_length is None or delta >= min_gap_length):
                    result.append( { 'start': last_end_time, 'end': segment_start, 'gap': True } )
            
            last_end_time = segment_end
            result.append(segment)

        # Also include total duration if specified
        if (total_duration is not None and last_end_time < total_duration):
            delta = total_duration - segment_start

            if (min_gap_length is None or delta >= min_gap_length):
                result.append( { 'start': last_end_time, 'end': total_duration, 'gap': True } )

        return result

    # Expand the end time of each segment to the start of the next segment
    def expand_gaps(self, segments: List[Dict[str, Any]], total_duration: float):
        result = []

        if len(segments) == 0:
            return result

        # Add gap at the beginning if needed
        if (segments[0]['start'] > 0):
            result.append({ 'start': 0, 'end': segments[0]['start'], 'gap': True } )

        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            delta = next_segment['start'] - current_segment['end']

            # Expand if the gap actually exists
            if (delta >= 0):
                current_segment = current_segment.copy()
                current_segment['expand_amount'] = delta
                current_segment['end'] = next_segment['start']
            
            result.append(current_segment)

        last_segment = result[-1]

        # Also include total duration if specified
        if (total_duration is not None):
            last_segment = result[-1]

            if (last_segment['end'] < total_duration):
                last_segment = last_segment.copy()
                last_segment['end'] = total_duration
                result[-1] = last_segment

        return result

    def adjust_whisper_timestamp(self, segments: Iterator[dict], adjust_seconds: float, max_source_time: float = None):
        result = []

        for segment in segments:
            segment_start = float(segment['start'])
            segment_end = float(segment['end'])

            # Filter segments?
            if (max_source_time is not None):
                if (segment_start > max_source_time):
                    continue
                segment_end = min(max_source_time, segment_end)

                new_segment = segment.copy()

            # Add to start and end
            new_segment['start'] = segment_start + adjust_seconds
            new_segment['end'] = segment_end + adjust_seconds
            result.append(new_segment)
        return result

    def pad_timestamps(self, timestamps: List[Dict[str, Any]], padding_left: float, padding_right: float):
        if (padding_left == 0 and padding_right == 0):
            return timestamps
        result = []

        for i in range(len(timestamps)):
            prev_entry = timestamps[i - 1] if i > 0 else None
            curr_entry = timestamps[i]
            next_entry = timestamps[i + 1] if i < len(timestamps) - 1 else None

            segment_start = curr_entry['start']
            segment_end = curr_entry['end']

            if padding_left is not None:
                segment_start = max(prev_entry['end'] if prev_entry else 0, segment_start - padding_left)
            if padding_right is not None:
                segment_end = segment_end + padding_right

                # Do not pad past the next segment
                if (next_entry is not None):
                    segment_end = min(next_entry['start'], segment_end)

            result.append({ 'start': segment_start, 'end': segment_end })

        return result

    def merge_timestamps(self, timestamps: List[Dict[str, Any]], max_merge_gap: float, max_merge_size: float):
        if max_merge_gap is None:
            return timestamps

        result = []
        current_entry = None

        for entry in timestamps:
            if current_entry is None:
                current_entry = entry
                continue

            # Get distance to the previous entry
            distance = entry['start'] - current_entry['end']
            current_entry_size = current_entry['end'] - current_entry['start']

            if distance <= max_merge_gap and (max_merge_size is None or current_entry_size <= max_merge_size):
                # Merge
                current_entry['end'] = entry['end']
            else:
                # Output current entry
                result.append(current_entry)
                current_entry = entry
        
        # Add final entry
        if current_entry is not None:
            result.append(current_entry)

        return result

    def multiply_timestamps(self, timestamps: List[Dict[str, Any]], factor: float):
        result = []

        for entry in timestamps:
            start = entry['start']
            end = entry['end']

            result.append({
                'start': start * factor,
                'end': end * factor
            })
        return result

class VadSileroTranscription(AbstractTranscription):
    def __init__(self, segment_padding_left=SEGMENT_PADDING_LEFT, segment_padding_right=SEGMENT_PADDING_RIGHT, 
                 max_silent_period=MAX_SILENT_PERIOD, max_merge_size=MAX_MERGE_SIZE, transcribe_non_speech: bool = False, 
                 copy = None):
        super().__init__(segment_padding_left=segment_padding_left, segment_padding_right=segment_padding_right, 
                         max_silent_period=max_silent_period, max_merge_size=max_merge_size, transcribe_non_speech=transcribe_non_speech)

        if copy:
            self.model = copy.model
            self.get_speech_timestamps = copy.get_speech_timestamps
        else:
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            (self.get_speech_timestamps, _, _, _, _) = utils

    def get_transcribe_timestamps(self, audio: str):
        wav = self.get_audio_segment(audio)

        sample_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate, threshold=SPEECH_TRESHOLD)
        seconds_timestamps = self.multiply_timestamps(sample_timestamps, factor=1 / self.sampling_rate) 

        return seconds_timestamps

# A very simple VAD that just marks every N seconds as speech
class VadPeriodicTranscription(AbstractTranscription):
    def __init__(self, periodic_duration: int):
        super().__init__()
        self.periodic_duration = periodic_duration

    def get_transcribe_timestamps(self, audio: str):
        # Get duration in seconds
        audio_duration = float(ffmpeg.probe(audio)["format"]["duration"])
        result = []

        # Generate a timestamp every N seconds
        start_timestamp = 0

        while (start_timestamp < audio_duration):
            end_timestamp = min(start_timestamp + self.periodic_duration, audio_duration)
            segment_duration = end_timestamp - start_timestamp

            # Minimum duration is 1 second
            if (segment_duration >= 1):
                result.append( {  'start': start_timestamp, 'end': end_timestamp } )

            start_timestamp = end_timestamp

        return result

def load_audio(file: str, sample_rate: int = 16000, 
               start_time: str = None, duration: str = None):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    start_time: str
        The start time, using the standard FFMPEG time duration syntax, or None to disable.
    
    duration: str
        The duration, using the standard FFMPEG time duration syntax, or None to disable.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        inputArgs = {'threads': 0}

        if (start_time is not None):
            inputArgs['ss'] = start_time
        if (duration is not None):
            inputArgs['t'] = duration

        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, **inputArgs)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0