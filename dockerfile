FROM huggingface/transformers-pytorch-gpu
EXPOSE 7860

ADD . /opt/whisper-webui/
RUN python3 -m pip install -r /opt/whisper-webui/requirements.txt

# Note: Models will be downloaded on demand to the directory /root/.cache/whisper.
# You can also bind this directory in the container to somewhere on the host.

WORKDIR /opt/whisper-webui/
ENTRYPOINT ["python3"]
CMD ["app-network.py"]