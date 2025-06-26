
# Arabic Multi-Speaker TTS API

A FastAPI service that converts Arabic text into speech using a pre-trained multi-speaker TTS model. Each request can select a speaker gender (male/female) and choose between CPU or GPU (if available).

## Features

- **Arabic TTS**: High-quality speech synthesis from Arabic text.
- **Speaker Gender**: Choose Male or Female voice.
- **Per-Request Device Selection**: Optionally specify device: "cpu" or device: "cuda" in each /tts/ call (defaults to GPU if available, otherwise CPU).
- **Lightweight Streaming**: Audio is streamed back as WAV—no temporary files.

## Requirements

- Docker (optional, for containerization)
- Python 3.10+
- FastAPI
- PyTorch (with CUDA support for GPU inference)
- phonemizer
- scipy

## Getting Started

1. Clone the repo

```bash
git clone https://github.com/your-username/arabic-tts-api.git
cd arabic-tts-api
```

2. Prepare your models/ folder

Place the following files under `./models/`:

- `config.yml`
- One or more `.pth` checkpoints (the latest will be loaded)
- `ref_audioM.wav` (male reference)
- `ref_audioF.wav` (female reference)

Your directory should look like:

```
arabic-tts-api/
├─ app.py
├─ inferenceMSP.py
├─ models/
│  ├─ config.yml
│  ├─ model_epoch_10.pth
│  ├─ ref_audioM.wav
│  └─ ref_audioF.wav
└─ requirements.txt
```

3. (Optional) Build & Run with Docker

```bash
# build
docker build -t arabic-tts-api .

# run with GPU
docker run --gpus all -p 8000:8000 arabic-tts-api

# or run on CPU only
docker run -p 8000:8000 arabic-tts-api
```

4. Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

**Endpoint**: `GET /`

**Response**:

```json
{ 
  "message": "Arabic MSP-TTS up. POST /tts/ to get audio." 
}
```

### Generate Speech

**Endpoint**: `POST /tts/`

**Headers**:

- Content-Type: application/json
- Accept: audio/wav

**Request Body**:

```json
{
  "text":           "<Arabic text here>",
  "speaker_gender": "Male" | "Female",
  "device":         "cpu"  | "cuda"      // optional
}
```

Omitting `device` → defaults to `cuda` if GPU is available, otherwise `cpu`.

**Response**:

- Status: 200 OK
- WAV audio stream in the body
- Header `X-Device` indicates which device ran inference

**Example: Male Voice on GPU**

```bash
curl -X POST "http://localhost:8000/tts/"      -H "Content-Type: application/json"      -d '{
       "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
       "speaker_gender": "Male",
       "device": "cuda"
     }'      --output male.wav
```

**Example: Female Voice on CPU**

```bash
curl -X POST "http://localhost:8000/tts/"      -H "Content-Type: application/json"      -d '{
       "text": "أهلاً بكم في خدمة تحويل النص إلى كلام",
       "speaker_gender": "Female",
       "device": "cpu"
     }'      --output female.wav
```

### Notes

- The service pre-loads the latest checkpoint on both CPU and GPU (if available) at startup for minimal latency.
- Ensure your models/config.yml matches the architecture of your checkpoint.
- Remove or rename extra .pth files if you want to control which checkpoint is used.

### Troubleshooting

- **500 Internal Server Error** → Verify that `config.yml` and at least one `.pth` file are present in `./models/`.
- **Missing reference audio** → Make sure `ref_audioM.wav` and `ref_audioF.wav` exist in the working directory.
- **GPU not detected** → Confirm CUDA and PyTorch GPU support are installed and that an NVIDIA GPU is available.
