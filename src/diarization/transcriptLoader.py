
import json
from pathlib import Path

def load_transcript_json(transcript_file: str):
    """
    Parse a Whisper JSON file into a Whisper JSON object

    # Parameters:
    transcript_file (str): Path to the Whisper JSON file
    """
    with open(transcript_file, "r", encoding="utf-8") as f:
        whisper_result = json.load(f)

    # Format of Whisper JSON file:
    #  {
    # "text": " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    # "segments": [
    #    {
    #        "text": " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    #        "start": 0.0,
    #        "end": 10.36,
    #        "words": [
    #            {
    #                "start": 0.0,
    #                "end": 0.56,
    #                "word": " And",
    #                "probability": 0.61767578125
    #            },
    #            {
    #                "start": 0.56,
    #                "end": 0.88,
    #                "word": " so",
    #                "probability": 0.9033203125
    #            },
    # etc.  

    return whisper_result


def load_transcript_srt(subtitle_file: str):
    import srt

    """
    Parse a SRT file into a Whisper JSON object

    # Parameters:
    subtitle_file (str): Path to the SRT file
    """
    with open(subtitle_file, "r", encoding="utf-8") as f:
        subs = srt.parse(f)

        whisper_result = {
            "text": "",
            "segments": []
        }

        for sub in subs:
            # Subtitle(index=1, start=datetime.timedelta(seconds=33, microseconds=843000), end=datetime.timedelta(seconds=38, microseconds=97000), content='地球上只有3%的水是淡水', proprietary='')
            segment = {
                "text": sub.content,
                "start": sub.start.total_seconds(),
                "end": sub.end.total_seconds(),
                "words": []
            }
            whisper_result["segments"].append(segment)
            whisper_result["text"] += sub.content

        return whisper_result

def load_transcript(file: str):
    # Determine file type
    file_extension = Path(file).suffix.lower()

    if file_extension == ".json":
        return load_transcript_json(file)
    elif file_extension == ".srt":
        return load_transcript_srt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")