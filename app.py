from typing import Iterator

from io import StringIO
import os
import pathlib
import tempfile

# External programs
import whisper
import ffmpeg

# UI
import gradio as gr

from src.download import ExceededMaximumDuration, download_url
from src.utils import slugify, write_srt, write_vtt
from src.vad import NonSpeechStrategy, VadPeriodicTranscription, VadSileroTranscription

# Limitations (set to -1 to disable)
DEFAULT_INPUT_AUDIO_MAX_DURATION = 600 # seconds

# Whether or not to automatically delete all uploaded files, to save disk space
DELETE_UPLOADED_FILES = True

# Gradio seems to truncate files without keeping the extension, so we need to truncate the file prefix ourself 
MAX_FILE_PREFIX_LENGTH = 17

LANGUAGES = [ 
 "English", "Chinese", "German", "Spanish", "Russian", "Korean", 
 "French", "Japanese", "Portuguese", "Turkish", "Polish", "Catalan", 
 "Dutch", "Arabic", "Swedish", "Italian", "Indonesian", "Hindi", 
 "Finnish", "Vietnamese", "Hebrew", "Ukrainian", "Greek", "Malay", 
 "Czech", "Romanian", "Danish", "Hungarian", "Tamil", "Norwegian", 
 "Thai", "Urdu", "Croatian", "Bulgarian", "Lithuanian", "Latin", 
 "Maori", "Malayalam", "Welsh", "Slovak", "Telugu", "Persian", 
 "Latvian", "Bengali", "Serbian", "Azerbaijani", "Slovenian", 
 "Kannada", "Estonian", "Macedonian", "Breton", "Basque", "Icelandic", 
 "Armenian", "Nepali", "Mongolian", "Bosnian", "Kazakh", "Albanian",
 "Swahili", "Galician", "Marathi", "Punjabi", "Sinhala", "Khmer", 
 "Shona", "Yoruba", "Somali", "Afrikaans", "Occitan", "Georgian", 
 "Belarusian", "Tajik", "Sindhi", "Gujarati", "Amharic", "Yiddish", 
 "Lao", "Uzbek", "Faroese", "Haitian Creole", "Pashto", "Turkmen", 
 "Nynorsk", "Maltese", "Sanskrit", "Luxembourgish", "Myanmar", "Tibetan",
 "Tagalog", "Malagasy", "Assamese", "Tatar", "Hawaiian", "Lingala", 
 "Hausa", "Bashkir", "Javanese", "Sundanese"
]

class WhisperTranscriber:
    def __init__(self, inputAudioMaxDuration: float = DEFAULT_INPUT_AUDIO_MAX_DURATION, deleteUploadedFiles: bool = DELETE_UPLOADED_FILES):
        self.model_cache = dict()

        self.vad_model = None
        self.inputAudioMaxDuration = inputAudioMaxDuration
        self.deleteUploadedFiles = deleteUploadedFiles

    def transcribe_webui(self, modelName, languageName, urlData, uploadFile, microphoneData, task, vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow):
        try:
            source, sourceName = self.__get_source(urlData, uploadFile, microphoneData)
            
            try:
                selectedLanguage = languageName.lower() if len(languageName) > 0 else None
                selectedModel = modelName if modelName is not None else "base"

                model = self.model_cache.get(selectedModel, None)
                
                if not model:
                    model = whisper.load_model(selectedModel)
                    self.model_cache[selectedModel] = model

                # Execute whisper
                result = self.transcribe_file(model, source, selectedLanguage, task, vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)

                # Write result
                downloadDirectory = tempfile.mkdtemp()
                
                filePrefix = slugify(sourceName, allow_unicode=True)
                download, text, vtt = self.write_result(result, filePrefix, downloadDirectory)

                return download, text, vtt

            finally:
                # Cleanup source
                if self.deleteUploadedFiles:
                    print("Deleting source file " + source)
                    os.remove(source)
        
        except ExceededMaximumDuration as e:
            return [], ("[ERROR]: Maximum remote video length is " + str(e.maxDuration) + "s, file was " + str(e.videoDuration) + "s"), "[ERROR]"

    def transcribe_file(self, model: whisper.Whisper, audio_path: str, language: str, task: str = None, vad: str = None, 
                        vadMergeWindow: float = 5, vadMaxMergeSize: float = 150, vadPadding: float = 1, vadPromptWindow: float = 1, **decodeOptions: dict):
        # Callable for processing an audio file
        whisperCallable = lambda audio, prompt : model.transcribe(audio, language=language, task=task, initial_prompt=prompt, **decodeOptions)

        # The results
        if (vad == 'silero-vad'):
            # Silero VAD where non-speech gaps are transcribed
            process_gaps = self._create_silero_vad(NonSpeechStrategy.CREATE_SEGMENT, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = process_gaps.transcribe(audio_path, whisperCallable)
        elif (vad == 'silero-vad-skip-gaps'):
            # Silero VAD where non-speech gaps are simply ignored
            skip_gaps = self._create_silero_vad(NonSpeechStrategy.SKIP, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = skip_gaps.transcribe(audio_path, whisperCallable)
        elif (vad == 'silero-vad-expand-into-gaps'):
            # Use Silero VAD where speech-segments are expanded into non-speech gaps
            expand_gaps = self._create_silero_vad(NonSpeechStrategy.EXPAND_SEGMENT, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = expand_gaps.transcribe(audio_path, whisperCallable)
        elif (vad == 'periodic-vad'):
            # Very simple VAD - mark every 5 minutes as speech. This makes it less likely that Whisper enters an infinite loop, but
            # it may create a break in the middle of a sentence, causing some artifacts.
            periodic_vad = VadPeriodicTranscription(periodic_duration=vadMaxMergeSize)
            result = periodic_vad.transcribe(audio_path, whisperCallable)
        else:
            # Default VAD
            result = whisperCallable(audio_path, None)

        return result

    def _create_silero_vad(self, non_speech_strategy: NonSpeechStrategy, vadMergeWindow: float = 5, vadMaxMergeSize: float = 150, vadPadding: float = 1, vadPromptWindow: float = 1):
        # Use Silero VAD 
        if (self.vad_model is None):
            self.vad_model = VadSileroTranscription()

        result = VadSileroTranscription(non_speech_strategy = non_speech_strategy, 
                max_silent_period=vadMergeWindow, max_merge_size=vadMaxMergeSize, 
                segment_padding_left=vadPadding, segment_padding_right=vadPadding, 
                max_prompt_window=vadPromptWindow, copy=self.vad_model)

        return result

    def write_result(self, result: dict, source_name: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        text = result["text"]
        language = result["language"]
        languageMaxLineWidth = self.__get_max_line_width(language)

        print("Max line width " + str(languageMaxLineWidth))
        vtt = self.__get_subs(result["segments"], "vtt", languageMaxLineWidth)
        srt = self.__get_subs(result["segments"], "srt", languageMaxLineWidth)

        output_files = []
        output_files.append(self.__create_file(srt, output_dir, source_name + "-subs.srt"));
        output_files.append(self.__create_file(vtt, output_dir, source_name + "-subs.vtt"));
        output_files.append(self.__create_file(text, output_dir, source_name + "-transcript.txt"));

        return output_files, text, vtt

    def clear_cache(self):
        self.model_cache = dict()
        self.vad_model = None

    def __get_source(self, urlData, uploadFile, microphoneData):
        if urlData:
            # Download from YouTube
            source = download_url(urlData, self.inputAudioMaxDuration)[0]
        else:
            # File input
            source = uploadFile if uploadFile is not None else microphoneData

            if self.inputAudioMaxDuration > 0:
                # Calculate audio length
                audioDuration = ffmpeg.probe(source)["format"]["duration"]
            
                if float(audioDuration) > self.inputAudioMaxDuration:
                    raise ExceededMaximumDuration(videoDuration=audioDuration, maxDuration=self.inputAudioMaxDuration, message="Video is too long")

        file_path = pathlib.Path(source)
        sourceName = file_path.stem[:MAX_FILE_PREFIX_LENGTH] + file_path.suffix

        return source, sourceName

    def __get_max_line_width(self, language: str) -> int:
        if (language and language.lower() in ["japanese", "ja", "chinese", "zh"]):
            # Chinese characters and kana are wider, so limit line length to 40 characters
            return 40
        else:
            # TODO: Add more languages
            # 80 latin characters should fit on a 1080p/720p screen
            return 80

    def __get_subs(self, segments: Iterator[dict], format: str, maxLineWidth: int) -> str:
        segmentStream = StringIO()

        if format == 'vtt':
            write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
        elif format == 'srt':
            write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
        else:
            raise Exception("Unknown format " + format)

        segmentStream.seek(0)
        return segmentStream.read()

    def __create_file(self, text: str, directory: str, fileName: str) -> str:
        # Write the text to a file
        with open(os.path.join(directory, fileName), 'w+', encoding="utf-8") as file:
            file.write(text)

        return file.name


def create_ui(inputAudioMaxDuration, share=False, server_name: str = None):
    ui = WhisperTranscriber(inputAudioMaxDuration)

    ui_description = "Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse " 
    ui_description += " audio and is also a multi-task model that can perform multilingual speech recognition "
    ui_description += " as well as speech translation and language identification. "

    ui_description += "\n\n\n\nFor longer audio files (>10 minutes), it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option."

    if inputAudioMaxDuration > 0:
        ui_description += "\n\n" + "Max audio file length: " + str(inputAudioMaxDuration) + " s"

    ui_article = "Read the [documentation here](https://huggingface.co/spaces/aadnk/whisper-webui/blob/main/docs/options.md)"

    demo = gr.Interface(fn=ui.transcribe_webui, description=ui_description, article=ui_article, inputs=[
        gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="medium", label="Model"),
        gr.Dropdown(choices=sorted(LANGUAGES), label="Language"),
        gr.Text(label="URL (YouTube, etc.)"),
        gr.Audio(source="upload", type="filepath", label="Upload Audio"), 
        gr.Audio(source="microphone", type="filepath", label="Microphone Input"),
        gr.Dropdown(choices=["transcribe", "translate"], label="Task"),
        gr.Dropdown(choices=["none", "silero-vad", "silero-vad-skip-gaps", "silero-vad-expand-into-gaps", "periodic-vad"], label="VAD"),
        gr.Number(label="VAD - Merge Window (s)", precision=0, value=5),
        gr.Number(label="VAD - Max Merge Size (s)", precision=0, value=30),
        gr.Number(label="VAD - Padding (s)", precision=None, value=1),
        gr.Number(label="VAD - Prompt Window (s)", precision=None, value=3)
    ], outputs=[
        gr.File(label="Download"),
        gr.Text(label="Transcription"), 
        gr.Text(label="Segments")
    ])

    demo.launch(share=share, server_name=server_name)   

if __name__ == '__main__':
    create_ui(DEFAULT_INPUT_AUDIO_MAX_DURATION)