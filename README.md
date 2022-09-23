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

To run this program locally, first install Python 3.9+ and Git. Then install Pytorch 10.1 and all the dependencies:
```
pip install -r requirements.txt
```

Finally, run the full version (no audio length restrictions) of the app:
```
python app-full.py
```