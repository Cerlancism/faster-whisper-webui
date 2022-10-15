# Options
To transcribe or translate an audio file, you can either copy an URL from a website (all [websites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) 
supported by YT-DLP will work, including YouTube). Otherwise, upload an audio file (choose "All Files (*.*)" 
in the file selector to select any file type, including video files) or use the microphone.

For longer audio files (>10 minutes), it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option.

## Model
Select the model that Whisper will use to transcribe the audio:

| Size   | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|--------|------------|--------------------|--------------------|---------------|----------------|
| tiny   | 39 M       | tiny.en            | tiny               | ~1 GB         | ~32x           |
| base   | 74 M       | base.en            | base               | ~1 GB         | ~16x           |
| small  | 244 M      | small.en           | small              | ~2 GB         | ~6x            |
| medium | 769 M      | medium.en          | medium             | ~5 GB         | ~2x            |
| large  | 1550 M     | N/A                | large              | ~10 GB        | 1x             |

## Language

Select the language, or leave it empty for Whisper to automatically detect it. 

Note that if the selected language and the language in the audio differs, Whisper may start to translate the audio to the selected 
language. For instance, if the audio is in English but you select Japaneese, the model may translate the audio to Japanese.

## Inputs
The options "URL (YouTube, etc.)", "Upload Audio" or "Micriphone Input" allows you to send an audio input to the model.

Note that the UI will only process the first valid input - i.e. if you enter both an URL and upload an audio, it will only process 
the URL. 

## Task
Select the task - either "transcribe" to transcribe the audio to text, or "translate" to translate it to English.

## Vad
* none
  * Run whisper on the entire audio input
* silero-vad
   * Use Silero VAD to detect sections that contain speech, and run whisper on independently on each section. Whisper is also run 
     on the gaps between each speech section.
* silero-vad-skip-gaps
   * As above, but sections that doesn't contain speech according to Silero will be skipped. This will be slightly faster, but 
     may cause dialogue to be skipped.
* periodic-vad
   * Create sections of speech every 'VAD - Max Merge Size' seconds. This is very fast and simple, but will potentially break 
     a sentence or word in two.

## VAD - Merge Window
If set, any adjacent speech sections that are at most this number of seconds apart will be automatically merged.

## VAD - Max Merge Size (s)
Disables merging of adjacent speech sections if they are this number of seconds long.

## VAD - Padding (s)
The number of seconds (floating point) to add to the beginning and end of each speech section. Setting this to a number
larger than zero ensures that Whisper is more likely to correctly transcribe a sentence in the beginning of 
a speech section. However, this also increases the probability of Whisper assigning the wrong timestamp 
to each transcribed line. The default value is 1 second.