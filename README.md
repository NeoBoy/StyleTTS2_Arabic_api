
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
http://localhost:8000/
```

It should return a message like:

```json
{
  "message": "Arabic Multi-Speaker TTS API. Use POST /tts to generate speech."
}
```

### Step 5: Test the API



#### a. **Health Check (Check if the API is running)**

This curl command checks if your API is up and running.

```bash
curl -X GET "http://localhost:8000/"
```

**Expected Output**:

```json
{
  "message": "Arabic Multi-Speaker TTS API. Use POST /tts to generate speech."
}
```

---

#### b. **Switch Device (CPU/GPU)**

You can use this curl command to switch between **CPU** and **GPU**. By default, the API uses **GPU** if available, but you can explicitly change it.

- **Switch to CPU**:

```bash
curl -X POST "http://localhost:8000/set_device/" -H "Content-Type: application/json" -d '{
  "device": "cpu"
}'
```

- **Switch to GPU**:

```bash
curl -X POST "http://localhost:8000/set_device/" -H "Content-Type: application/json" -d '{
  "device": "cuda"
}'
```

**Expected Output** (for both CPU and GPU switches):

```json
{
  "message": "Device switched to cpu",  # or "cuda"
  "current_device": "cpu"  # or "cuda"
}
```

---

#### c. **Generate TTS for Male Speaker**

To generate speech for the Arabic text `"السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ"`, use the following curl command.

- **Male Voice (GPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
  "speaker_gender": "Male",
  "device": "cuda",
  "embedding_scale": 1.0
}' --output synthesized_audio_male.wav
```

- **Male Voice (CPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
  "speaker_gender": "Male",
  "device": "cpu",
  "embedding_scale": 1.0
}' --output synthesized_audio_male.wav
```

This command will generate the audio file for the **male** voice and save it as `synthesized_audio_male.wav`.

**Expected Output**:

The audio will be returned and saved in the specified `.wav` file (e.g., `synthesized_audio_male.wav`).

---

#### d. **Generate TTS for Female Speaker**

Similarly, you can generate speech for the **female** voice using the following commands.

- **Female Voice (GPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
  "speaker_gender": "Female",
  "device": "cuda",
  "embedding_scale": 1.0
}' --output synthesized_audio_female.wav
```

- **Female Voice (CPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
  "speaker_gender": "Female",
  "device": "cpu",
  "embedding_scale": 1.0
}' --output synthesized_audio_female.wav
```

This command will generate the audio file for the **female** voice and save it as `synthesized_audio_female.wav`.

**Expected Output**:

The audio will be returned and saved in the specified `.wav` file (e.g., `synthesized_audio_female.wav`).

---

#### e. **Example for Multiple Speaker Generation**

You can test both **male** and **female** voices using different curl commands. The difference is only in the `"speaker_gender"` field.

- **Male Voice (default GPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "أهلاً وسهلاً في API تحويل النص إلى كلام",
  "speaker_gender": "Male"
}' --output male_output.wav
```

- **Female Voice (default GPU)**:

```bash
curl -X POST "http://localhost:8000/tts/" -H "Content-Type: application/json" -d '{
  "text": "أهلاً وسهلاً في API تحويل النص إلى كلام",
  "speaker_gender": "Female"
}' --output female_output.wav
```

---

### Summary of curl Command Usage:

- **Check Health**: `GET /`
- **Switch Device**: `POST /set_device/`
- **Generate TTS**: `POST /tts/`
  - With options for **CPU/GPU** and **Male/Female** voice selection.
- **Save Generated Audio**: Use the `--output` flag to save the response file as `.wav`.
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
