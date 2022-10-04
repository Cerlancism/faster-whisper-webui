from collections import Counter
from dis import dis
from typing import Any, Iterator, List, Dict

from pprint import pprint
import torch

import ffmpeg
import numpy as np

SPEECH_TRESHOLD = 0.3
MAX_SILENT_PERIOD = 10 # seconds

SEGMENT_PADDING_LEFT = 1 # Start detected text segment early
SEGMENT_PADDING_RIGHT = 4 # End detected segments late

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

class VadTranscription:
    def __init__(self):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        (self.get_speech_timestamps, _, _, _, _) = utils

    def transcribe(self, audio: str, whisperCallable):
        SAMPLING_RATE = 16000
        wav = load_audio(audio, sample_rate=SAMPLING_RATE)

        # get speech timestamps from full audio file
        sample_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=SAMPLING_RATE, threshold=SPEECH_TRESHOLD)
        seconds_timestamps = self.convert_seconds(sample_timestamps, sampling_rate=SAMPLING_RATE)

        padded = self.pad_timestamps(seconds_timestamps, SEGMENT_PADDING_LEFT, SEGMENT_PADDING_RIGHT)
        merged = self.merge_timestamps(padded, MAX_SILENT_PERIOD)

        print("Timestamps:")
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
            segment_duration = segment['end'] - segment_start

            segment_audio = load_audio(audio, sample_rate=SAMPLING_RATE, start_time = str(segment_start) + "s", duration = str(segment_duration) + "s")

            print("Running whisper on " + str(segment_start) + ", duration: " + str(segment_duration))
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
        result = []

        for entry in timestamps:
            segment_start = entry['start']
            segment_end = entry['end']

            if padding_left is not None:
                segment_start = max(0, segment_start - padding_left)
            if padding_right is not None:
                segment_end = segment_end + padding_right

            result.append({ 'start': segment_start, 'end': segment_end })

        return result

    def merge_timestamps(self, timestamps: List[Dict[str, Any]], max_distance: float):
        result = []
        current_entry = None

        for entry in timestamps:
            if current_entry is None:
                current_entry = entry
                continue

            # Get distance to the previous entry
            distance = entry['start'] - current_entry['end']

            if distance <= max_distance:
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

    def convert_seconds(self, timestamps: List[Dict[str, Any]], sampling_rate: int):
        result = []

        for entry in timestamps:
            start = entry['start']
            end = entry['end']

            result.append({
                'start': start / sampling_rate,
                'end': end / sampling_rate
            })
        return result
        