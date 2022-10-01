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

Finally, run the full version (no audio length restrictions) of the app:
```
python app-full.py
```

# Docker

To run it in Docker, first install Docker and optionally the NVIDIA Container Toolkit in order to use the GPU. Then 
check out this repository and build an image:
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