# Multi-Language TTS API (Arabic & English)

A FastAPI service that converts Arabic and English text into speech using pre-trained StyleTTS2 models. Each request can select language, speaker gender (male/female), choose between CPU or GPU for optimal performance, and optionally provide custom reference audio for voice cloning.

## Features

- **Multi-Language Support**: High-quality speech synthesis for Arabic and English text
- **Speaker Gender Selection**: Choose Male or Female voice for both languages
- **Flexible Device Selection**: Per-request device selection ("cpu" or "cuda") with automatic fallback
- **Custom Voice Cloning**: Upload your own reference audio to clone any voice style
- **Default Reference Audio**: Built-in high-quality reference voices when no custom audio provided
- **Intelligent Text Processing**: Automatic text chunking for long texts to avoid model limitations
- **Streaming Audio**: Audio is streamed back as WAV with no temporary files
- **Comprehensive Error Handling**: Robust validation and error reporting
- **Performance Monitoring**: Response headers include inference time and device usage

## Supported Languages

- **Arabic**: Native StyleTTS2 model with Arabic PLBERT
- **English**: StyleTTS2 model with English PLBERT
- **Device Support**: Both languages support CPU and CUDA acceleration
- **Voice Cloning**: Both languages support custom reference audio upload

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
git clone https://github.com/NeoBoy/StyleTTS2_Arabic_api.git multilang_tts_api
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
  "endpoints": {
    "tts": "POST /tts/ - Generate speech from text (supports both JSON and form data with optional reference audio)",
    "health": "GET / - API status"
  },
  "usage_example": {
    "default_audio": {
      "method": "Form data",
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
    },
    "custom_audio": {
      "method": "Form data with file upload",
      "note": "Add reference_audio file to clone voice style"
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

**Content-Type**: `multipart/form-data` (for file upload support)

**Parameters**:
- `text` (required): Input text in the specified language
- `speaker_gender` (required): Voice gender selection (`Male` or `Female`)
- `language` (optional): Target language (`arabic` or `english`, defaults to `arabic`)
- `device` (optional): Device selection (`cpu` or `cuda`, auto-detects if not specified)
- `reference_audio` (optional): Custom reference audio file for voice cloning

**⚠️ Important Note About Voice Cloning**:
When `reference_audio` is provided, the voice characteristics are determined **entirely by the uploaded reference audio file**. The `speaker_gender` parameter is **ignored in this case** because the voice style, gender, and all vocal characteristics come from your custom reference audio.

- **Default Voice Mode** (no `reference_audio`): `speaker_gender` selects between pre-trained male/female voices
- **Voice Cloning Mode** (with `reference_audio`): Voice characteristics come from your uploaded file, `speaker_gender` is ignored

**Default Reference Audio**:
- **Arabic Male**: Built-in Arabic male reference (`ref_audioM.wav`)
- **Arabic Female**: Built-in Arabic female reference (`ref_audioF.wav`)
- **English Male**: Built-in English male reference
- **English Female**: Built-in English female reference

**Custom Reference Audio**:
- Upload any audio file (WAV, MP3, etc.) to clone that voice style
- Supports common audio formats
- If no file is provided, uses default references based on language and gender

**Response**:
- Status: 200 OK
- Content-Type: audio/wav
- Response Headers:
  - `X-Device`: Device used for inference
  - `X-Language`: Language processed
  - `X-Speaker-Gender`: Gender used
  - `X-Inference-Time`: Processing time
  - `X-Reference-Type`: "default" or "custom"
  - `X-Reference-File`: Reference file name used

## Usage Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/"
curl -X GET "http://localhost:8000/health"
```

### Default Reference Audio

**Arabic Text-to-Speech (Male Voice)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ" \
  -F "speaker_gender=Male" \
  -F "language=arabic" \
  --output arabic_male.wav
```

**Arabic Text-to-Speech (Female Voice, CPU)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=أهلاً وسهلاً بكم في خدمة تحويل النص إلى كلام" \
  -F "speaker_gender=Female" \
  -F "language=arabic" \
  -F "device=cpu" \
  --output arabic_female_cpu.wav
```

**English Text-to-Speech (Female Voice, CUDA)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=Hello, welcome to our advanced text-to-speech service" \
  -F "speaker_gender=Female" \
  -F "language=english" \
  -F "device=cuda" \
  --output english_female_cuda.wav
```

**English Text-to-Speech (Male Voice, Default Device)**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=This is a demonstration of our English text-to-speech capabilities" \
  -F "speaker_gender=Male" \
  -F "language=english" \
  --output english_male.wav
```

### Custom Reference Audio (Voice Cloning)

**Arabic with Custom Voice**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=مرحبا بكم في خدمة التحويل من النص إلى الكلام" \
  -F "speaker_gender=Male" \
  -F "language=arabic" \
  -F "reference_audio=@my_arabic_voice.wav" \
  --output arabic_custom_voice.wav
```

**English with Custom Voice**:
```bash
curl -X POST "http://localhost:8000/tts/" \
  -F "text=Hello, this will sound exactly like my voice" \
  -F "speaker_gender=Female" \
  -F "language=english" \
  -F "device=cuda" \
  -F "reference_audio=@target_voice.wav" \
  --output english_cloned_voice.wav
```

### Python Examples

**Default Reference Audio**:
```python
import requests

# Arabic TTS with default reference
arabic_data = {
    "text": "مرحباً بكم في خدمة التحويل من النص إلى الكلام",
    "speaker_gender": "Female",
    "language": "arabic",
    "device": "cuda"
}

response = requests.post("http://localhost:8000/tts/", data=arabic_data)
if response.status_code == 200:
    with open("arabic_output.wav", "wb") as f:
        f.write(response.content)
    print(f"Device used: {response.headers.get('X-Device')}")
    print(f"Reference type: {response.headers.get('X-Reference-Type')}")
    print(f"Inference time: {response.headers.get('X-Inference-Time')}")

# English TTS with default reference
english_data = {
    "text": "Welcome to our multilingual text-to-speech service",
    "speaker_gender": "Male",
    "language": "english"
}

response = requests.post("http://localhost:8000/tts/", data=english_data)
if response.status_code == 200:
    with open("english_output.wav", "wb") as f:
        f.write(response.content)
```

**Custom Reference Audio (Voice Cloning)**:
```python
import requests

# Voice cloning with custom reference audio
with open("my_voice_sample.wav", "rb") as ref_file:
    files = {"reference_audio": ref_file}
    data = {
        "text": "This text will be spoken in my cloned voice",
        "speaker_gender": "Male",
        "language": "english",
        "device": "cuda"
    }
    
    response = requests.post("http://localhost:8000/tts/", files=files, data=data)
    
    if response.status_code == 200:
        with open("cloned_voice_output.wav", "wb") as f:
            f.write(response.content)
        print(f"Reference type: {response.headers.get('X-Reference-Type')}")
        print(f"Reference file: {response.headers.get('X-Reference-File')}")
        print(f"Inference time: {response.headers.get('X-Inference-Time')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
```

## Performance Benchmarks

Based on test results:

| Language | Device | Gender | Reference Type | Avg. Inference Time | Performance Boost |
|----------|--------|---------|----------------|-------------------|------------------|
| Arabic   | CUDA   | Male    | Default        | ~0.5s            | 4-5x faster     |
| Arabic   | CUDA   | Female  | Default        | ~0.3s            | 4-5x faster     |
| Arabic   | CUDA   | Any     | Custom         | ~0.6s            | 4-5x faster     |
| Arabic   | CPU    | Male    | Default        | ~3.5s            | Baseline        |
| Arabic   | CPU    | Female  | Default        | ~2.5s            | Baseline        |
| English  | CUDA   | Male    | Default        | ~0.3s            | 6-8x faster     |
| English  | CUDA   | Female  | Default        | ~0.3s            | 6-8x faster     |
| English  | CUDA   | Any     | Custom         | ~0.4s            | 6-8x faster     |
| English  | CPU    | Male    | Default        | ~2.3s            | Baseline        |
| English  | CPU    | Female  | Default        | ~2.3s            | Baseline        |

## Features & Limitations

### ✅ Supported Features
- Multi-language support (Arabic & English)
- GPU acceleration with automatic fallback
- Real-time streaming response
- Voice cloning with custom reference audio
- Default high-quality reference voices
- Automatic temporary file cleanup
- Comprehensive error handling
- Performance monitoring

### ⚠️ Current Limitations
- Maximum recommended text length: ~400 characters per chunk
- Custom reference audio should be clear speech samples (3-10 seconds recommended)
- Supports common audio formats (WAV, MP3, FLAC, etc.)
- CUDA memory requirements for larger models
- Voice cloning quality depends on reference audio quality

### 📋 Voice Cloning Best Practices
- Use clean, clear audio samples (minimal background noise)
- 3-10 seconds of speech is usually sufficient
- Single speaker recordings work best
- Consistent audio quality throughout the reference
- WAV format recommended for best results
- **Remember**: `speaker_gender` parameter is ignored when using custom reference audio
- **Voice characteristics** (including gender, accent, tone) come entirely from your reference file
- **Test your reference audio** first to ensure it produces the desired voice characteristics

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
- Default and custom reference audio
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

**Custom Reference Audio Issues**:
- Ensure uploaded file is a valid audio format
- Check file size (very large files may cause timeouts)
- Verify audio quality and clarity

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

## API Response Examples

### Successful Response Headers
```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Device: cuda
X-Language: arabic
X-Speaker-Gender: Male
X-Inference-Time: 0.45s
X-Reference-Type: custom
X-Reference-File: my_voice_sample.wav
Content-Disposition: attachment; filename=tts_arabic_Male_custom.wav
```

### Error Response Example
```json
{
  "detail": "Language 'french' not available. Available languages: ['arabic', 'english']"
}
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