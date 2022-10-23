from abc import ABC, abstractmethod
from collections import Counter, deque

from typing import Any, Deque, Iterator, List, Dict

from pprint import pprint

from src.segments import merge_timestamps

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
from enum import Enum

class NonSpeechStrategy(Enum):
    """
    Ignore non-speech frames segments.
    """
    SKIP = 1
    """
    Just treat non-speech segments as speech.
    """
    CREATE_SEGMENT = 2
    """
    Expand speech segments into subsequent non-speech segments.
    """
    EXPAND_SEGMENT = 3

# Defaults for Silero
SPEECH_TRESHOLD = 0.3

# Minimum size of segments to process
MIN_SEGMENT_DURATION = 1

# The maximum time for texts from old segments to be used in the next segment 
MAX_PROMPT_WINDOW = 0 # seconds (0 = disabled)
PROMPT_NO_SPEECH_PROB = 0.1 # Do not pass the text from segments with a no speech probability higher than this

VAD_MAX_PROCESSING_CHUNK = 60 * 60 # 60 minutes of audio

class TranscriptionConfig(ABC):
    def __init__(self, non_speech_strategy: NonSpeechStrategy = NonSpeechStrategy.SKIP, 
                       segment_padding_left: float = None, segment_padding_right = None, max_silent_period: float = None, 
                       max_merge_size: float = None, max_prompt_window: float = None):
        self.non_speech_strategy = non_speech_strategy
        self.segment_padding_left = segment_padding_left
        self.segment_padding_right = segment_padding_right
        self.max_silent_period = max_silent_period
        self.max_merge_size = max_merge_size
        self.max_prompt_window = max_prompt_window

class PeriodicTranscriptionConfig(TranscriptionConfig):
    def __init__(self, periodic_duration: float, non_speech_strategy: NonSpeechStrategy = NonSpeechStrategy.SKIP, 
                       segment_padding_left: float = None, segment_padding_right = None, max_silent_period: float = None, 
                       max_merge_size: float = None, max_prompt_window: float = None):
        super().__init__(non_speech_strategy, segment_padding_left, segment_padding_right, max_silent_period, max_merge_size, max_prompt_window)
        self.periodic_duration = periodic_duration

class AbstractTranscription(ABC):
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate

    def get_audio_segment(self, str, start_time: str = None, duration: str = None):
        return load_audio(str, self.sampling_rate, start_time, duration)

    @abstractmethod
    def get_transcribe_timestamps(self, audio: str, config: TranscriptionConfig):
        """
        Get the start and end timestamps of the sections that should be transcribed by this VAD method.

        Parameters
        ----------
        audio: str
            The audio file.
        config: TranscriptionConfig
            The transcription configuration.

        Returns
        -------
        A list of start and end timestamps, in fractional seconds.
        """
        return 

    def transcribe(self, audio: str, whisperCallable, config: TranscriptionConfig):
        """
        Transcribe the given audo file.

        Parameters
        ----------
        audio: str
            The audio file.

        whisperCallable: Callable[[Union[str, np.ndarray, torch.Tensor], int, str, str], dict[str, Union[dict, Any]]]
            The callback that is used to invoke Whisper on an audio file/buffer. The first parameter is the audio file/buffer, 
            the second parameter is an optional text prompt, and the last is the current detected language. The return value is the result of the Whisper call.

        Returns
        -------
        A list of start and end timestamps, in fractional seconds.
        """

        # get speech timestamps from full audio file
        seconds_timestamps = self.get_transcribe_timestamps(audio, config)

        #for seconds_timestamp in seconds_timestamps:
        #    print("VAD timestamp ", format_timestamp(seconds_timestamp['start']), " to ", format_timestamp(seconds_timestamp['end']))

        merged = merge_timestamps(seconds_timestamps, config.max_silent_period, config.max_merge_size, config.segment_padding_left, config.segment_padding_right)

        # A deque of transcribed segments that is passed to the next segment as a prompt
        prompt_window = deque()

        print("Timestamps:")
        pprint(merged)

        if config.non_speech_strategy != NonSpeechStrategy.SKIP:
            max_audio_duration = get_audio_duration(audio)

            # Expand segments to include the gaps between them
            if (config.non_speech_strategy == NonSpeechStrategy.CREATE_SEGMENT):
                # When we have a prompt window, we create speech segments betwen each segment if we exceed the merge size
                merged = self.fill_gaps(merged, total_duration=max_audio_duration, max_expand_size=config.max_merge_size)
            elif config.non_speech_strategy == NonSpeechStrategy.EXPAND_SEGMENT: 
                # With no prompt window, it is better to just expand the segments (this effectively passes the prompt to the next segment)
                merged = self.expand_gaps(merged, total_duration=max_audio_duration)
            else:
                raise Exception("Unknown non-speech strategy: " + str(config.non_speech_strategy))

            print("Transcribing non-speech:")
            pprint(merged)

        result = {
            'text': "",
            'segments': [],
            'language': ""
        }
        languageCounter = Counter()
        detected_language = None

        segment_index = -1

        # For each time segment, run whisper
        for segment in merged:
            segment_index += 1
            segment_start = segment['start']
            segment_end = segment['end']
            segment_expand_amount = segment.get('expand_amount', 0)
            segment_gap = segment.get('gap', False)

            segment_duration = segment_end - segment_start

            if segment_duration < MIN_SEGMENT_DURATION:
                continue;

            # Audio to run on Whisper
            segment_audio = self.get_audio_segment(audio, start_time = str(segment_start), duration = str(segment_duration))
            # Previous segments to use as a prompt
            segment_prompt = ' '.join([segment['text'] for segment in prompt_window]) if len(prompt_window) > 0 else None
    
            # Detected language
            detected_language = languageCounter.most_common(1)[0][0] if len(languageCounter) > 0 else None

            print("Running whisper from ", format_timestamp(segment_start), " to ", format_timestamp(segment_end), ", duration: ", 
                  segment_duration, "expanded: ", segment_expand_amount, "prompt: ", segment_prompt, "language: ", detected_language)
            segment_result = whisperCallable(segment_audio, segment_index, segment_prompt, detected_language)

            adjusted_segments = self.adjust_timestamp(segment_result["segments"], adjust_seconds=segment_start, max_source_time=segment_duration)

            # Propagate expand amount to the segments
            if (segment_expand_amount > 0):
                segment_without_expansion = segment_duration - segment_expand_amount

                for adjusted_segment in adjusted_segments:
                    adjusted_segment_end = adjusted_segment['end']

                    # Add expand amount if the segment got expanded
                    if (adjusted_segment_end > segment_without_expansion):
                        adjusted_segment["expand_amount"] = adjusted_segment_end - segment_without_expansion

            # Append to output
            result['text'] += segment_result['text']
            result['segments'].extend(adjusted_segments)

            # Increment detected language
            if not segment_gap:
                languageCounter[segment_result['language']] += 1

            # Update prompt window
            self.__update_prompt_window(prompt_window, adjusted_segments, segment_end, segment_gap, config)
            
        if detected_language is not None:
            result['language'] = detected_language

        return result
            
    def __update_prompt_window(self, prompt_window: Deque, adjusted_segments: List, segment_end: float, segment_gap: bool, config: TranscriptionConfig):
        if (config.max_prompt_window is not None and config.max_prompt_window > 0):
            # Add segments to the current prompt window (unless it is a speech gap)
            if not segment_gap:
                for segment in adjusted_segments:
                    if segment.get('no_speech_prob', 0) <= PROMPT_NO_SPEECH_PROB:
                        prompt_window.append(segment)

            while (len(prompt_window) > 0):
                first_end_time = prompt_window[0].get('end', 0)
                # Time expanded in the segments should be discounted from the prompt window
                first_expand_time = prompt_window[0].get('expand_amount', 0)

                if (first_end_time - first_expand_time < segment_end - config.max_prompt_window):
                    prompt_window.popleft()
                else:
                    break

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

        # Add last segment
        last_segment = segments[-1]
        result.append(last_segment)

        # Also include total duration if specified
        if (total_duration is not None):
            last_segment = result[-1]

            if (last_segment['end'] < total_duration):
                last_segment = last_segment.copy()
                last_segment['end'] = total_duration
                result[-1] = last_segment

        return result

    def fill_gaps(self, segments: List[Dict[str, Any]], total_duration: float, max_expand_size: float = None):
        result = []

        if len(segments) == 0:
            return result

        # Add gap at the beginning if needed
        if (segments[0]['start'] > 0):
            result.append({ 'start': 0, 'end': segments[0]['start'], 'gap': True } )

        for i in range(len(segments) - 1):
            expanded = False
            current_segment = segments[i]
            next_segment = segments[i + 1]

            delta = next_segment['start'] - current_segment['end']

            if (max_expand_size is not None and delta <= max_expand_size):
                # Just expand the current segment
                current_segment = current_segment.copy()
                current_segment['expand_amount'] = delta
                current_segment['end'] = next_segment['start']
                expanded = True

            result.append(current_segment)

            # Add a gap to the next segment if needed
            if (delta >= 0 and not expanded):
                result.append({ 'start': current_segment['end'], 'end': next_segment['start'], 'gap': True } )
            
        # Add last segment
        last_segment = segments[-1]
        result.append(last_segment)

        # Also include total duration if specified
        if (total_duration is not None):
            last_segment = result[-1]

            delta = total_duration - last_segment['end']

            if (delta > 0):
                if (max_expand_size is not None and delta <= max_expand_size):
                    # Expand the last segment
                    last_segment = last_segment.copy()
                    last_segment['expand_amount'] = delta
                    last_segment['end'] = total_duration
                    result[-1] = last_segment
                else:
                    result.append({ 'start': last_segment['end'], 'end': total_duration, 'gap': True } )

        return result

    def adjust_timestamp(self, segments: Iterator[dict], adjust_seconds: float, max_source_time: float = None):
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
    def __init__(self, sampling_rate: int = 16000):
        super().__init__(sampling_rate=sampling_rate)

        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, _, _, _) = utils


    def get_transcribe_timestamps(self, audio: str, config: TranscriptionConfig):
        audio_duration = get_audio_duration(audio)
        result = []

        # Divide procesisng of audio into chunks
        chunk_start = 0.0

        while (chunk_start < audio_duration):
            chunk_duration = min(audio_duration - chunk_start, VAD_MAX_PROCESSING_CHUNK)

            print("Processing VAD in chunk from {} to {}".format(format_timestamp(chunk_start), format_timestamp(chunk_start + chunk_duration)))
            wav = self.get_audio_segment(audio, str(chunk_start), str(chunk_duration))

            sample_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate, threshold=SPEECH_TRESHOLD)
            seconds_timestamps = self.multiply_timestamps(sample_timestamps, factor=1 / self.sampling_rate) 
            adjusted = self.adjust_timestamp(seconds_timestamps, adjust_seconds=chunk_start, max_source_time=chunk_start + chunk_duration)

            #pprint(adjusted)

            result.extend(adjusted)
            chunk_start += chunk_duration

        return result

# A very simple VAD that just marks every N seconds as speech
class VadPeriodicTranscription(AbstractTranscription):
    def __init__(self, sampling_rate: int = 16000):
        super().__init__(sampling_rate=sampling_rate)

    def get_transcribe_timestamps(self, audio: str, config: PeriodicTranscriptionConfig):
        # Get duration in seconds
        audio_duration = get_audio_duration(audio)
        result = []

        # Generate a timestamp every N seconds
        start_timestamp = 0

        while (start_timestamp < audio_duration):
            end_timestamp = min(start_timestamp + config.periodic_duration, audio_duration)
            segment_duration = end_timestamp - start_timestamp

            # Minimum duration is 1 second
            if (segment_duration >= 1):
                result.append( {  'start': start_timestamp, 'end': end_timestamp } )

            start_timestamp = end_timestamp

        return result

def get_audio_duration(file: str):
    return float(ffmpeg.probe(file)["format"]["duration"])

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