# Multi-Language TTS API (Arabic & English)

A FastAPI service that converts Arabic and English text into speech using pre-trained StyleTTS2 models. Each request can select language, speaker gender (male/female), and choose between CPU or GPU for optimal performance.

## Features

- **Multi-Language Support**: High-quality speech synthesis for Arabic and English text
- **Speaker Gender Selection**: Choose Male or Female voice for both languages
- **Flexible Device Selection**: Per-request device selection ("cpu" or "cuda") with automatic fallback
- **Intelligent Text Processing**: Automatic text chunking for long texts to avoid model limitations
- **Streaming Audio**: Audio is streamed back as WAV with no temporary files
- **Comprehensive Error Handling**: Robust validation and error reporting
- **Performance Monitoring**: Response headers include inference time and device usage

## Supported Languages

- **Arabic**: Native StyleTTS2 model with Arabic PLBERT
- **English**: StyleTTS2 model with English PLBERT
- **Device Support**: Both languages support CPU and CUDA acceleration

## Requirements

- Docker (recommended for easy deployment)
- Python 3.10+
- FastAPI
- PyTorch (with CUDA support for GPU inference)
- phonemizer (with espeak backend)
- scipy, librosa, numpy

## Getting Started

### Option 1: Build from Source

```bash
# Clone the repository
git clone https://github.com/NeoBoy/arabic-tts-api.git multilang_tts_api
cd multilang_tts_api

# Build Docker image
docker build -t multilang_tts_api .

# Run with GPU support
docker run --gpus all -p 8000:8000 multilang_tts_api

# Run CPU only
docker run -p 8000:8000 multilang_tts_api
```

### Option 2: Use Pre-built Image

Download the pre-built image from Google Drive: [http://tiny.cc/arabicTTS]

```bash
# Load the pre-built image
docker load -i multilang_tts_api.tar

# Run with GPU support
docker run --gpus all -p 8000:8000 multilang_tts_api:latest

# Run CPU only
docker run -p 8000:8000 multilang_tts_api:latest
```

## API Endpoints

### Health Check

**Endpoint**: `GET /`

**Response**:
```json
{
  "message": "Multi-language TTS API",
  "description": "Arabic and English Text-to-Speech service",
  "available_languages": ["arabic", "english"],
  "supported_genders": ["Male", "Female"],
  "total_model_instances": 4,
  "device_info": {
    "arabic": "CPU and CUDA supported",
    "english": "CPU and CUDA supported"
  },
  "endpoints": {
    "tts": "POST /tts/ - Generate speech from text",
    "health": "GET / - API status"
  },
  "usage_example": {
    "arabic": {
      "text": "مرحبا بك في خدمة التحويل من النص إلى الكلام",
      "speaker_gender": "Male",
      "language": "arabic"
    },
    "english": {
      "text": "Hello, welcome to our text-to-speech service",
      "speaker_gender": "Female",
      "language": "english"
    }
  }
}
```

### Health Status

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "available_languages": ["arabic", "english"],
  "cuda_available": true,
  "devices": ["cpu", "cuda"]
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
  "text": "<Text in Arabic or English>",
  "speaker_gender": "Male" | "Female",
  "language": "arabic" | "english",
  "device": "cpu" | "cuda"  // optional
}
```

**Parameters**:
- `text`: Input text in the specified language
- `speaker_gender`: Voice gender selection (`Male` or `Female`)
- `language`: Target language (`arabic` or `english`)
- `device`: Optional device selection (defaults to `cuda` if available, otherwise `cpu`)

**Response**:
- Status: 200 OK
- Content-Type: audio/wav
- Response Headers:
  - `X-Device`: Device used for inference
  - `X-Language`: Language processed
  - `X-Speaker-Gender`: Gender used
  - `X-Inference-Time`: Processing time
  - `X-Text-Chunks`: Number of text chunks processed

## Usage Examples

### Arabic Text-to-Speech

**Male Voice (Default Device)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
    "speaker_gender": "Male",
    "language": "arabic"
  }' \
  --output arabic_male.wav
```

**Female Voice (CPU)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "أهلاً وسهلاً بكم في خدمة تحويل النص إلى كلام",
    "speaker_gender": "Female",
    "language": "arabic",
    "device": "cpu"
  }' \
  --output arabic_female_cpu.wav
```

### English Text-to-Speech

**Female Voice (CUDA)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to our advanced text-to-speech service",
    "speaker_gender": "Female",
    "language": "english",
    "device": "cuda"
  }' \
  --output english_female_cuda.wav
```

**Male Voice (Default Device)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a demonstration of our English text-to-speech capabilities",
    "speaker_gender": "Male",
    "language": "english"
  }' \
  --output english_male.wav
```

### Python Example

```python
import requests

# Arabic TTS
arabic_payload = {
    "text": "مرحباً بكم في خدمة التحويل من النص إلى الكلام",
    "speaker_gender": "Female",
    "language": "arabic",
    "device": "cuda"
}

response = requests.post("http://localhost:8000/tts/", json=arabic_payload)
if response.status_code == 200:
    with open("arabic_output.wav", "wb") as f:
        f.write(response.content)
    print(f"Device used: {response.headers.get('X-Device')}")
    print(f"Inference time: {response.headers.get('X-Inference-Time')}")

# English TTS
english_payload = {
    "text": "Welcome to our multilingual text-to-speech service",
    "speaker_gender": "Male",
    "language": "english"
}

response = requests.post("http://localhost:8000/tts/", json=english_payload)
if response.status_code == 200:
    with open("english_output.wav", "wb") as f:
        f.write(response.content)
```

## Performance Benchmarks

Based on test results:

| Language | Device | Gender | Avg. Inference Time | Performance Boost |
|----------|--------|---------|-------------------|------------------|
| Arabic   | CUDA   | Male    | ~0.5s            | 4-5x faster     |
| Arabic   | CUDA   | Female  | ~0.3s            | 4-5x faster     |
| Arabic   | CPU    | Male    | ~3.5s            | Baseline        |
| Arabic   | CPU    | Female  | ~2.5s            | Baseline        |
| English  | CUDA   | Male    | ~0.3s            | 6-8x faster     |
| English  | CUDA   | Female  | ~0.3s            | 6-8x faster     |
| English  | CPU    | Male    | ~2.3s            | Baseline        |
| English  | CPU    | Female  | ~2.3s            | Baseline        |

## Features & Limitations

### ✅ Supported Features
- Multi-language support (Arabic & English)
- GPU acceleration with automatic fallback
- Real-time streaming response
- Automatic text chunking for long texts
- Comprehensive error handling
- Performance monitoring

### ⚠️ Current Limitations
- Maximum recommended text length: ~400 characters per chunk
- Requires specific reference audio files for each gender
- English female voice uses fallback reference audio
- CUDA memory requirements for larger models

## File Structure

```
multilang_tts_api/
├── app.py                  # Main FastAPI application
├── models/
│   ├── config_ar.yml      # Arabic model configuration
│   ├── config_en.yml      # English model configuration
│   ├── model_ar.pth       # Arabic model weights
│   ├── model_en.pth       # English model weights
│   └── reference_audio/   # English reference audio files
├── ref_audioM.wav         # Arabic male reference
├── ref_audioF.wav         # Arabic female reference
├── Utils/                 # Model utilities
├── test_api.py           # Comprehensive API tests
└── Dockerfile            # Container configuration
```

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will test:
- Health endpoints
- Both languages with all gender/device combinations
- Error handling and edge cases
- Performance and concurrent requests
- Large text processing

## Troubleshooting

### Common Issues

**500 Internal Server Error**:
- Verify model files exist in `./models/` directory
- Check that configuration files are valid YAML
- Ensure reference audio files are present

**Missing Reference Audio**:
- Arabic: Ensure `ref_audioM.wav` and `ref_audioF.wav` exist
- English: Check `models/reference_audio/` directory

**GPU Not Detected**:
- Confirm CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `torch.cuda.is_available()`
- Use `--gpus all` flag when running Docker

**Text Length Errors**:
- Keep individual texts under 400 characters
- The API automatically chunks longer texts
- Very long texts may require manual segmentation

**Language Not Available**:
- Check startup logs for model loading errors
- Verify both model files and configs are present
- Test with `/health` endpoint to see available languages

### Debug Mode

For development, run with detailed logging:

```bash
docker run --gpus all -p 8000:8000 -e DEBUG=1 multilang_tts_api
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python test_api.py`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- StyleTTS2 for the base TTS architecture
- PLBERT for multilingual text encoding
- FastAPI for the web framework