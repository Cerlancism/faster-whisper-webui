import os 

from tempfile import mkdtemp
from yt_dlp import YoutubeDL
from yt_dlp.postprocessor import PostProcessor

class FilenameCollectorPP(PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information["filepath"])
        return [], information

def downloadUrl(url: str, maxDuration: int = None):
    destinationDirectory = mkdtemp()

    ydl_opts = {
        "format": "bestaudio/best",
        'playlist_items': '1',
        'paths': {
            'home': destinationDirectory
        }
    }
    filename_collector = FilenameCollectorPP()

    with YoutubeDL(ydl_opts) as ydl:
        if maxDuration and maxDuration > 0:
            info = ydl.extract_info(url, download=False)
            duration = info['duration']

            if duration >= maxDuration:
                raise ExceededMaximumDuration(videoDuration=duration, maxDuration=maxDuration, message="Video is too long")

        ydl.add_post_processor(filename_collector)
        ydl.download([url])

    if len(filename_collector.filenames) <= 0:
        raise Exception("Cannot download " + url)

    result = filename_collector.filenames[0]
    print("Downloaded " + result)

    return result 


class ExceededMaximumDuration(Exception):
    def __init__(self, videoDuration, maxDuration, message):
        self.videoDuration = videoDuration
        self.maxDuration = maxDuration
        super().__init__(message)