FROM nvcr.io/nvidia/tritonserver:23.10-py3
RUN pip install --ignore-installed git+https://github.com/coqui-ai/TTS.git transformers==4.37.1 deepspeed google-cloud-storage