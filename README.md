---
title: Whisper Webui
emoji: ⚡
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 3.3.1
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Running Locally

To run this program locally, first install Python 3.9+ and Git. Then install Pytorch 10.1+ and all the other dependencies:
```
pip install -r requirements.txt
```

Finally, run the full version (no audio length restrictions) of the app with parallel CPU/GPU enabled:
```
python app.py --input_audio_max_duration -1 --server_name 127.0.0.1 --auto_parallel True
```

You can also run the CLI interface, which is similar to Whisper's own CLI but also supports the following additional arguments:
```
python cli.py \
[--vad {none,silero-vad,silero-vad-skip-gaps,silero-vad-expand-into-gaps,periodic-vad}] \
[--vad_merge_window VAD_MERGE_WINDOW] \
[--vad_max_merge_size VAD_MAX_MERGE_SIZE] \
[--vad_padding VAD_PADDING] \
[--vad_prompt_window VAD_PROMPT_WINDOW]
[--vad_cpu_cores NUMBER_OF_CORES]
[--vad_parallel_devices COMMA_DELIMITED_DEVICES]
[--auto_parallel BOOLEAN]
```
In addition, you may also use URL's in addition to file paths as input.
```
python cli.py --model large --vad silero-vad --language Japanese "https://www.youtube.com/watch?v=4cICErqqRSM"
```

## Parallel Execution

You can also run both the Web-UI or the CLI on multiple GPUs in parallel, using the `vad_parallel_devices` option. This takes a comma-delimited list of 
device IDs (0, 1, etc.) that Whisper should be distributed to and run on concurrently:
```
python cli.py --model large --vad silero-vad --language Japanese \
--vad_parallel_devices 0,1 "https://www.youtube.com/watch?v=4cICErqqRSM"
```

Note that this requires a VAD to function properly, otherwise only the first GPU will be used. Though you could use `period-vad` to avoid taking the hit
of running Silero-Vad, at a slight cost to accuracy.

This is achieved by creating N child processes (where N is the number of selected devices), where Whisper is run concurrently. In `app.py`, you can also 
set the `vad_process_timeout` option. This configures the number of seconds until a process is killed due to inactivity, freeing RAM and video memory. 
The default value is 30 minutes.

```
python app.py --input_audio_max_duration -1 --vad_parallel_devices 0,1 --vad_process_timeout 3600
```

To execute the Silero VAD itself in parallel, use the `vad_cpu_cores` option:
```
python app.py --input_audio_max_duration -1 --vad_parallel_devices 0,1 --vad_process_timeout 3600 --vad_cpu_cores 4
```

You may also use `vad_process_timeout` with a single device (`--vad_parallel_devices 0`), if you prefer to always free video memory after a period of time.

### Auto Parallel

You can also set `auto_parallel` to `True`. This will set `vad_parallel_devices` to use all the GPU devices on the system, and `vad_cpu_cores` to be equal to the number of
cores (up to 8):
```
python app.py --input_audio_max_duration -1 --auto_parallel True
```

# Docker

To run it in Docker, first install Docker and optionally the NVIDIA Container Toolkit in order to use the GPU. 
Then either use the GitLab hosted container below, or check out this repository and build an image:
```
sudo docker build -t whisper-webui:1 .
```

You can then start the WebUI with GPU support like so:
```
sudo docker run -d --gpus=all -p 7860:7860 whisper-webui:1
```

Leave out "--gpus=all" if you don't have access to a GPU with enough memory, and are fine with running it on the CPU only:
```
sudo docker run -d -p 7860:7860 whisper-webui:1
```

# GitLab Docker Registry

This Docker container is also hosted on GitLab:

```
sudo docker run -d --gpus=all -p 7860:7860 registry.gitlab.com/aadnk/whisper-webui:latest
```

## Custom Arguments

You can also pass custom arguments to `app.py` in the Docker container, for instance to be able to use all the GPUs in parallel:
```
sudo docker run -d --gpus all -p 7860:7860 \
--mount type=bind,source=/home/administrator/.cache/whisper,target=/root/.cache/whisper \
--restart=on-failure:15 registry.gitlab.com/aadnk/whisper-webui:latest \
app.py --input_audio_max_duration -1 --server_name 0.0.0.0 --vad_parallel_devices 0,1 \
--default_vad silero-vad --default_model_name large
```

You can also call `cli.py` the same way:
```
sudo docker run --gpus all \
--mount type=bind,source=/home/administrator/.cache/whisper,target=/root/.cache/whisper \
--mount type=bind,source=${PWD},target=/app/data \
registry.gitlab.com/aadnk/whisper-webui:latest \
cli.py --model large --vad_parallel_devices 0,1 --vad silero-vad \
--output_dir /app/data /app/data/YOUR-FILE-HERE.mp4
```

## Caching

Note that the models themselves are currently not included in the Docker images, and will be downloaded on the demand.
To avoid this, bind the directory /root/.cache/whisper to some directory on the host (for instance /home/administrator/.cache/whisper), where you can (optionally) 
prepopulate the directory with the different Whisper models. 
```
sudo docker run -d --gpus=all -p 7860:7860 \
--mount type=bind,source=/home/administrator/.cache/whisper,target=/root/.cache/whisper \
registry.gitlab.com/aadnk/whisper-webui:latest
```