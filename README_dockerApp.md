# Arabic MSP-TTS API

A high-performance FastAPI service for Arabic text-to-speech synthesis using the StyleTTS2 model. This API provides seamless integration for converting Arabic text to natural-sounding speech with configurable voice characteristics.

## Features

- 🎤 High-quality Arabic TTS synthesis using StyleTTS2
- 🔄 Multiple model support with dynamic loading
- 👥 Gender-specific voice synthesis (Male/Female)
- ⚡ Fast inference with optional diffusion modes
- 🐳 Docker containerization for easy deployment
- 📚 Interactive API documentation with OpenAPI/Swagger
- 🛡️ Input validation and error handling

## Architecture

```
arabic-msp-tts-api/
├── main.py                 # FastAPI application with TTS endpoints
├── inferenceMSP.py         # Core TTS inference pipeline
├── models/                 # TTS model artifacts
│   ├── config.yml          # Model configuration
│   ├── model15.pth         # Pre-trained model checkpoint
│   └── model24.pth         # Alternative model checkpoint
├── ref_audioM.wav          # Male speaker reference audio
├── ref_audioF.wav          # Female speaker reference audio
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build configuration
└── .dockerignore          # Docker build exclusions
```

## Prerequisites

### System Requirements
- Python 3.8+ (for local development)
- Docker 20.10+ (for containerized deployment)
- 4GB+ RAM (recommended for model inference)

### Dependencies
```bash
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
```

### Required Files
Ensure these files exist before starting:
- `models/config.yml` - Model configuration file
- `models/*.pth` - At least one trained model checkpoint
- `ref_audioM.wav` - Male speaker reference (24kHz, mono recommended)
- `ref_audioF.wav` - Female speaker reference (24kHz, mono recommended)

## Quick Start

### Local Development

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd arabic-msp-tts-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the development server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Verify installation**
Navigate to http://localhost:8000/docs to access the interactive API documentation.

### Docker Deployment

1. **Build the container image**
```bash
docker build -t arabic-msp-tts:latest .
```

2. **Run the containerized service**
```bash
docker run -d \
  --name arabic-tts-api \
  -p 8000:8000 \
  --restart unless-stopped \
  arabic-msp-tts:latest
```

3. **Health check**
```bash
curl http://localhost:8000/health
```

## API Reference

### Synthesis Endpoint

**`POST /synthesize`**

Generate Arabic speech from text input.

**Request Schema:**
```json
{
  "text": "string (required) - Arabic text to synthesize",
  "selected_model": "string (required) - Model filename from models/ directory",
  "speaker_gender": "string (required) - 'Male' or 'Female'",
  "generation_mode": "string (required) - Synthesis mode option"
}
```

**Generation Modes:**
- `"Compute style only (no diffusion)"` - Fast inference, good quality
- `"Diffusion conditioned on style"` - Slower inference, highest quality

**Response:**
- Content-Type: `audio/wav`
- Returns: Binary WAV audio data (24kHz, 16-bit, mono)

**Example Usage:**

```bash
# Basic synthesis request
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -H "Accept: audio/wav" \
  --data '{
    "text": "مرحباً بكم في خدمة تحويل النص إلى كلام",
    "selected_model": "model15.pth",
    "speaker_gender": "Female",
    "generation_mode": "Compute style only (no diffusion)"
  }' \
  --output "output.wav"
```

```python
# Python client example
import requests

response = requests.post(
    "http://localhost:8000/synthesize",
    json={
        "text": "النص العربي المراد تحويله إلى صوت",
        "selected_model": "model24.pth",
        "speaker_gender": "Male",
        "generation_mode": "Diffusion conditioned on style"
    }
)

if response.status_code == 200:
    with open("synthesized_audio.wav", "wb") as f:
        f.write(response.content)
```

### Models Endpoint

**`GET /models`**

Retrieve available TTS models.

**Response Schema:**
```json
{
  "models": ["model15.pth", "model24.pth"],
  "count": 2
}
```

**Example:**
```bash
curl http://localhost:8000/models
```

### Health Check

**`GET /health`**

Service health status endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "reference_files": ["ref_audioM.wav", "ref_audioF.wav"]
}
```

## Configuration

### Environment Variables

```bash
# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model settings
MODEL_PATH=./models
REFERENCE_AUDIO_PATH=.
DEFAULT_MODEL=model15.pth

# Performance tuning
MAX_TEXT_LENGTH=500
AUDIO_SAMPLE_RATE=24000  # Must match config.yml sr parameter
DEVICE=cuda              # 'cuda' or 'cpu'
```

### Model Configuration

The `models/config.yml` file contains comprehensive StyleTTS2 model parameters. Key sections include:

```yaml
# Core model architecture
model_params:
  multispeaker: false
  hidden_dim: 512
  n_token: 178          # Number of phoneme tokens
  n_mels: 80           # Mel spectrogram channels
  style_dim: 128       # Style vector dimensions
  max_dur: 50          # Maximum phoneme duration

# Audio preprocessing
preprocess_params:
  sr: 24000            # Sample rate
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

# Decoder configuration (HiFiGAN or iSTFTNet)
model_params:
  decoder:
    type: 'istftnet'
    upsample_rates: [10, 6]
    resblock_kernel_sizes: [3, 7, 11]

# Style diffusion model
model_params:
  diffusion:
    transformer:
      num_layers: 3
      num_heads: 8
      head_features: 64
```

**Important Configuration Notes:**
- `n_token: 178` must match your phoneme vocabulary size
- `sr: 24000` sets the audio sample rate (ensure reference audio matches)
- `multispeaker: false` for single-speaker models
- Decoder type can be `'hifigan'` or `'istftnet'`

## Performance Optimization

### Inference Speed
- Use `"Compute style only (no diffusion)"` for faster synthesis
- Consider model size vs. quality trade-offs
- Enable GPU acceleration if available

### Memory Management
- Monitor memory usage with multiple concurrent requests
- Implement request queuing for high-load scenarios
- Consider model caching strategies

### Scaling Recommendations
```bash
# Production deployment with multiple workers
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# Or use gunicorn for better process management
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Verify model files exist and are readable
ls -la models/
file models/*.pth
```

**Audio Quality Issues**
- Ensure reference audio files are 24kHz, mono WAV format (matching config.yml sr parameter)
- Check for text encoding issues (UTF-8 required)
- Verify model compatibility with input text
- Confirm phoneme tokenization matches n_token setting (178)

**Performance Problems**
- Monitor CPU/GPU utilization during inference
- Check available memory for large models
- Consider reducing concurrent request limits

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --log-level debug
```

### Validation Commands

```bash
# Test model availability
curl http://localhost:8000/models

# Validate synthesis pipeline
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"اختبار","selected_model":"model15.pth","speaker_gender":"Male","generation_mode":"Compute style only (no diffusion)"}' \
  --output test.wav && file test.wav
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)  
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **StyleTTS2 Arabic**: Base TTS implementation from [StyleTTS2_Arabic](https://github.com/NeoBoy/StyleTTS2_Arabic)
- **FastAPI**: Modern web framework for building APIs with Python
- **Uvicorn**: Lightning-fast ASGI server implementation
- **PyTorch**: Deep learning framework powering the TTS models

## Support

For issues and questions:
- 🐛 Report bugs via [GitHub Issues](link-to-issues)
- 💬 Join discussions in [GitHub Discussions](link-to-discussions)  
- 📧 Contact: [your-email@domain.com](mailto:your-email@domain.com)

---

*Last updated: June 2025*