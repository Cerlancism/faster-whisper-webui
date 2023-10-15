import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, List
import torch

import ffmpeg

class DiarizationEntry:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker

    def __repr__(self):
        return f"<DiarizationEntry start={self.start} end={self.end} speaker={self.speaker}>"
    
    def toJson(self):
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker
        }

class Diarization:
    def __init__(self, auth_token=None):
        if auth_token is None:
            auth_token = os.environ.get("HK_ACCESS_TOKEN")
            if auth_token is None:
                raise ValueError("No HuggingFace API Token provided - please use the --auth_token argument or set the HK_ACCESS_TOKEN environment variable")
        
        self.auth_token = auth_token
        self.initialized = False
        self.pipeline = None

    @staticmethod
    def has_libraries():
        try:
            import pyannote.audio
            import intervaltree
            return True
        except ImportError:
            return False

    def initialize(self):
        if self.initialized:
            return
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.auth_token)
        self.initialized = True

        # Load GPU mode if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print("Diarization - using GPU")
            self.pipeline = self.pipeline.to(torch.device(0))
        else:
            print("Diarization - using CPU")

    def run(self, audio_file, **kwargs):
        self.initialize()
        audio_file_obj = Path(audio_file)

        # Supported file types in soundfile is WAV, FLAC, OGG and MAT
        if audio_file_obj.suffix in [".wav", ".flac", ".ogg", ".mat"]:
            target_file = audio_file
        else:
            # Create temp WAV file
            target_file = tempfile.mktemp(prefix="diarization_", suffix=".wav")
            try:
                ffmpeg.input(audio_file).output(target_file, ac=1).run()
            except ffmpeg.Error as e:
                print(f"Error occurred during audio conversion: {e.stderr}")

        diarization = self.pipeline(target_file, **kwargs)

        if target_file != audio_file:
            # Delete temp file
            os.remove(target_file)

        # Yield result
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield DiarizationEntry(turn.start, turn.end, speaker)
    
    def mark_speakers(self, diarization_result: List[DiarizationEntry], whisper_result: dict):
        from intervaltree import IntervalTree
        result = whisper_result.copy()

        # Create an interval tree from the diarization results
        tree = IntervalTree()
        for entry in diarization_result:
            tree[entry.start:entry.end] = entry

        # Iterate through each segment in the Whisper JSON
        for segment in result["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]

            # Find overlapping speakers using the interval tree
            overlapping_speakers = tree[segment_start:segment_end]

            # If no speakers overlap with this segment, skip it
            if not overlapping_speakers:
                continue

            # If multiple speakers overlap with this segment, choose the one with the longest duration
            longest_speaker = None
            longest_duration = 0
            
            for speaker_interval in overlapping_speakers:
                overlap_start = max(speaker_interval.begin, segment_start)
                overlap_end = min(speaker_interval.end, segment_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > longest_duration:
                    longest_speaker = speaker_interval.data.speaker
                    longest_duration = overlap_duration

            # Add speakers
            segment["longest_speaker"] = longest_speaker
            segment["speakers"] = list([speaker_interval.data.toJson() for speaker_interval in overlapping_speakers])

            # The write_srt will use the longest_speaker if it exist, and add it to the text field

        return result

def _write_file(input_file: str, output_path: str, output_extension: str, file_writer: lambda f: None):
    if input_file is None:
        raise ValueError("input_file is required")
    if file_writer is None:
        raise ValueError("file_writer is required")

     # Write file
    if output_path is None:
        effective_path = os.path.splitext(input_file)[0] + "_output" + output_extension
    else:
        effective_path = output_path

    with open(effective_path, 'w+', encoding="utf-8") as f:
        file_writer(f)

    print(f"Output saved to {effective_path}")

def main():
    from src.utils import write_srt
    from src.diarization.transcriptLoader import load_transcript

    parser = argparse.ArgumentParser(description='Add speakers to a SRT file or Whisper JSON file using pyannote/speaker-diarization.')
    parser.add_argument('audio_file', type=str, help='Input audio file')
    parser.add_argument('whisper_file', type=str, help='Input Whisper JSON/SRT file')
    parser.add_argument('--output_json_file', type=str, default=None, help='Output JSON file (optional)')
    parser.add_argument('--output_srt_file', type=str, default=None, help='Output SRT file (optional)')
    parser.add_argument('--auth_token', type=str, default=None, help='HuggingFace API Token (optional)')
    parser.add_argument("--max_line_width", type=int, default=40, help="Maximum line width for SRT file (default: 40)")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers")

    args = parser.parse_args()

    print("\nReading whisper JSON from " + args.whisper_file)

    # Read whisper JSON or SRT file
    whisper_result = load_transcript(args.whisper_file)

    diarization = Diarization(auth_token=args.auth_token)
    diarization_result = list(diarization.run(args.audio_file, num_speakers=args.num_speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers))

    # Print result
    print("Diarization result:")
    for entry in diarization_result:
        print(f"  start={entry.start:.1f}s stop={entry.end:.1f}s speaker_{entry.speaker}")

    marked_whisper_result = diarization.mark_speakers(diarization_result, whisper_result)

    # Write output JSON to file
    _write_file(args.whisper_file, args.output_json_file, ".json", 
                lambda f: json.dump(marked_whisper_result, f, indent=4, ensure_ascii=False))

    # Write SRT
    _write_file(args.whisper_file, args.output_srt_file, ".srt", 
                lambda f: write_srt(marked_whisper_result["segments"], f, maxLineWidth=args.max_line_width))

if __name__ == "__main__":
    main()
    
    #test = Diarization()
    #print("Initializing")
    #test.initialize()

    #input("Press Enter to continue...")