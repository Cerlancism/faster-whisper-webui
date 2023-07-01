import unittest
import numpy as np
import sys

sys.path.append('../whisper-webui')
#print("Sys path: " + str(sys.path))

from src.whisper.abstractWhisperContainer import LambdaWhisperCallback
from src.vad import AbstractTranscription, TranscriptionConfig, VadSileroTranscription

class TestVad(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVad, self).__init__(*args, **kwargs)
        self.transcribe_calls = []

    def test_transcript(self):
        mock = MockVadTranscription(mock_audio_length=120)
        config = TranscriptionConfig()

        self.transcribe_calls.clear()
        result = mock.transcribe("mock", LambdaWhisperCallback(lambda segment, _1, _2, _3, _4: self.transcribe_segments(segment)), config)

        self.assertListEqual(self.transcribe_calls, [ 
            [30, 30],
            [100, 100]
        ])

        self.assertListEqual(result['segments'],
            [{'end': 50.0, 'start': 40.0, 'text': 'Hello world '},
            {'end': 120.0, 'start': 110.0, 'text': 'Hello world '}]
        )

    def transcribe_segments(self, segment):
        self.transcribe_calls.append(segment.tolist())

        # Dummy text
        return {
            'text': "Hello world ",
            'segments': [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "text": "Hello world "
                }   
            ],
            'language': ""
        }

class MockVadTranscription(AbstractTranscription):
    def __init__(self, mock_audio_length: float = 1000):
        super().__init__()
        self.mock_audio_length = mock_audio_length

    def get_audio_segment(self, str, start_time: str = None, duration: str = None):
        start_time_seconds = float(start_time.removesuffix("s"))
        duration_seconds = float(duration.removesuffix("s"))

        # For mocking, this just returns a simple numppy array
        return np.array([start_time_seconds, duration_seconds], dtype=np.float64)

    def get_transcribe_timestamps(self, audio: str, config: TranscriptionConfig, start_time: float, duration: float):
        result = []

        result.append( {  'start': 30, 'end': 60 } )
        result.append( {  'start': 100, 'end': 200 } )
        return result
        
    def get_audio_duration(self, audio: str, config: TranscriptionConfig):
        return self.mock_audio_length

if __name__ == '__main__':
    unittest.main()