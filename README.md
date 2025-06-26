
# Arabic Multi-Speaker TTS API

This repository provides a FastAPI-based Arabic Text-to-Speech (TTS) API using a pre-trained multi-speaker style TTS model. The model generates Arabic speech from input text, allowing users to select the speaker's gender (male/female) and provides a choice to switch between CPU or GPU devices.

## Features

- **Arabic TTS**: Generate Arabic speech from input text.
- **Speaker Gender Selection**: Choose between male or female speakers.
- **Device Selection**: Supports switching between CPU and GPU.
- **FastAPI-based REST API**: Easily deployable and can be tested with simple HTTP requests.

## Requirements

- Docker (for containerization)
- Python 3.7 or higher
- FastAPI
- PyTorch (with CUDA support for GPU)

## Setup Instructions

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/arabic-tts-api.git
cd arabic-tts-api
```

### Step 2: Build the Docker Image

Ensure that you have **Docker** installed on your machine.

To build the Docker image for this API, run the following command from the root of the cloned repository:

```bash
docker build -t arabic-tts-api .
```

### Step 3: Run the Docker Container

After building the Docker image, you can run the container. Use the following command to start the API service in the Docker container:

```bash
docker run --gpus all -p 8000:8000 -it arabic-tts-api
```

This command will run the FastAPI app, and it will be accessible at `http://localhost:8000`.

- If you don't have a GPU or don't want to use it, you can remove the `--gpus all` flag and run it on CPU.

### Step 4: Access the API

Once the API is running, you can access it at `http://localhost:8000`. You can check the status of the API by visiting:

```
GET http://localhost:8000/
```

It should return a message like:

```json
{
  "message": "Arabic Multi-Speaker TTS API. Use POST /tts to generate speech."
}
```

### Step 5: Test the API

#### Switch Device (CPU/GPU)

You can switch the device between **CPU** and **GPU** using the `/set_device/` endpoint. By default, the API selects **GPU** if available. To switch devices, use:

```bash
curl -X 'POST'   'http://localhost:8000/set_device/'   -H 'Content-Type: application/json'   -d '{
  "device": "cpu"
}'
```

#### Generate TTS (Text-to-Speech)

To generate TTS, send a POST request to `/tts/` with the desired text and speaker gender. Here's an example using `curl` to generate speech for the Arabic text "السَّلاَمُ عَلَيْكُمْ":

```bash
curl -X 'POST'   'http://localhost:8000/tts/'   -H 'Content-Type: application/json'   -d '{
  "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
  "speaker_gender": "Male",
  "embedding_scale": 1.0
}'
```

This will return a `.wav` file as a response.

#### Testing in Google Colab (Optional):

If you're testing in Google Colab, you can use the following Python code to send a request to your running API:

```python
import requests
import json

data = {
    "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
    "speaker_gender": "Male",
    "embedding_scale": 1.0
}

response = requests.post("http://localhost:8000/tts/", json=data)

with open("synthesized_audio.wav", "wb") as f:
    f.write(response.content)
```

This will save the response as `synthesized_audio.wav`, which can be played using any audio player.



## Notes

- Ensure the **model files** (`model24.pth`, `config.yml`, `ref_audioM.wav`, `ref_audioF.wav`) are placed in the `/models` directory. 
- If you want to change the model, update the `MODEL_DIR` to reflect the correct location of your model and configuration files.
- The `app.py` file includes **model warm-up** and **device switching** features, allowing seamless integration for both CPU and GPU.

## Troubleshooting

- If you encounter any errors related to missing files (e.g., `ref_audioM.wav`), ensure they are placed in the working directory or adjust paths accordingly.
- If the server is not starting correctly, check the logs for more information on the error.
