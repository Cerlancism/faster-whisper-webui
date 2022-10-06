import os 

from tempfile import mkdtemp
from tkinter import Y
from yt_dlp import YoutubeDL
import yt_dlp
from yt_dlp.postprocessor import PostProcessor

class FilenameCollectorPP(PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information["filepath"])
        return [], information

def downloadUrl(url: str, maxDuration: int = None):
    try:
        return _performDownload(url, maxDuration=maxDuration)
    except yt_dlp.utils.DownloadError as e:
        # In case of an OS error, try again with a different output template
        if e.msg and e.msg.find("[Errno 36] File name too long") >= 0:
            return _performDownload(url, maxDuration=maxDuration, outputTemplate="%(title).10s %(id)s.%(ext)s")
        pass

def _performDownload(url: str, maxDuration: int = None, outputTemplate: str = None):
    destinationDirectory = mkdtemp()

    ydl_opts = {
        "format": "bestaudio/best",
        'playlist_items': '1',
        'paths': {
            'home': destinationDirectory
        }
    }

    # Add output template if specified
    if outputTemplate:
        ydl_opts['outtmpl'] = outputTemplate

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