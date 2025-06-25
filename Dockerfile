# Universal Dockerfile for Arabic MSP-TTS API
# Compatible with: Docker, Fly.io, Railway, Render, Google Cloud Run, AWS ECS, Azure Container Instances
# Use PyTorch 2.4.1 with GPU support and CUDA 11.8
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    espeak-ng \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables (platform-agnostic)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set default port (can be overridden by platform)
ENV PORT=8000

# Clone StyleTTS2 Arabic repository
RUN git clone https://github.com/NeoBoy/StyleTTS2_Arabic.git /tmp/styletts2 && \
    cp -r /tmp/styletts2/* . && \
    rm -rf /tmp/styletts2

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install web framework and production dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart \
    aiofiles \
    gunicorn

# Create directory structure
RUN mkdir -p models Utils/JDC Utils/ASR Utils/PLBERT

# Download model files with comprehensive retry logic
RUN cd models && \
    echo "Downloading Arabic TTS models..." && \
    # Model 15
    for i in 1 2 3 4 5; do \
        echo "Downloading model15.pth (attempt $i/5)..." && \
        wget --timeout=180 --tries=2 --progress=dot:giga \
            -O model15.pth \
            "https://huggingface.co/blits-ai/style_tts2_finetune_audiobook4GPU/resolve/main/epoch_00015.pth" && \
        echo "model15.pth downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 10s before retry..." && sleep 10); \
    done && \
    # Model 24
    for i in 1 2 3 4 5; do \
        echo "Downloading model24.pth (attempt $i/5)..." && \
        wget --timeout=180 --tries=2 --progress=dot:giga \
            -O model24.pth \
            "https://huggingface.co/blits-ai/style_tts2_finetune_audiobook4GPU/resolve/main/epoch_00024.pth" && \
        echo "model24.pth downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 10s before retry..." && sleep 10); \
    done && \
    # Config file
    for i in 1 2 3; do \
        echo "Downloading config.yml (attempt $i/3)..." && \
        wget --timeout=60 --tries=2 \
            -O config.yml \
            "https://huggingface.co/fadi77/StyleTTS2-LibriTTS-arabic/resolve/main/config.yml" && \
        echo "config.yml downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 5s before retry..." && sleep 5); \
    done && \
    echo "All model files processed. Checking..." && \
    ls -la && \
    # Verify file sizes (basic integrity check)
    if [ -f "model15.pth" ] && [ $(stat -f%z "model15.pth" 2>/dev/null || stat -c%s "model15.pth") -gt 10000000 ]; then \
        echo "✓ model15.pth appears valid"; \
    else \
        echo "⚠ model15.pth may be incomplete"; \
    fi && \
    if [ -f "model24.pth" ] && [ $(stat -f%z "model24.pth" 2>/dev/null || stat -c%s "model24.pth") -gt 10000000 ]; then \
        echo "✓ model24.pth appears valid"; \
    else \
        echo "⚠ model24.pth may be incomplete"; \
    fi

# Download main.py from the repository (dockerApp directory)
RUN echo "Downloading main.py..." && \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=2 \
            -O main.py \
            "https://raw.githubusercontent.com/NeoBoy/StyleTTS2_Arabic/main/dockerApp/main.py" && \
        echo "main.py downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 5s before retry..." && sleep 5); \
    done

# Download inferenceMSP.py from the repository
RUN echo "Downloading inferenceMSP.py..." && \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=2 \
            -O inferenceMSP.py \
            "https://raw.githubusercontent.com/NeoBoy/StyleTTS2_Arabic/main/inferenceMSP.py" && \
        echo "inferenceMSP.py downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 5s before retry..." && sleep 5); \
    done

# Download reference audio files from the repository
RUN echo "Downloading reference audio files..." && \
    for i in 1 2 3; do \
        echo "Downloading ref_audioM.wav (attempt $i/3)..." && \
        wget --timeout=60 --tries=2 \
            -O ref_audioM.wav \
            "https://github.com/NeoBoy/StyleTTS2_Arabic/raw/main/ref_audioM.wav" && \
        echo "ref_audioM.wav downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 5s before retry..." && sleep 5); \
    done && \
    for i in 1 2 3; do \
        echo "Downloading ref_audioF.wav (attempt $i/3)..." && \
        wget --timeout=60 --tries=2 \
            -O ref_audioF.wav \
            "https://github.com/NeoBoy/StyleTTS2_Arabic/raw/main/ref_audioF.wav" && \
        echo "ref_audioF.wav downloaded successfully" && break || \
        (echo "Attempt $i failed, waiting 5s before retry..." && sleep 5); \
    done

# Create fallback reference audio files if downloads failed
RUN if [ ! -f "ref_audioM.wav" ] || [ $(stat -f%z "ref_audioM.wav" 2>/dev/null || stat -c%s "ref_audioM.wav") -lt 1000 ]; then \
        echo "Creating placeholder male reference audio..." && \
        sox -n -r 24000 -c 1 -b 16 ref_audioM.wav trim 0.0 2.0 synth sine 440; \
    fi && \
    if [ ! -f "ref_audioF.wav" ] || [ $(stat -f%z "ref_audioF.wav" 2>/dev/null || stat -c%s "ref_audioF.wav") -lt 1000 ]; then \
        echo "Creating placeholder female reference audio..." && \
        sox -n -r 24000 -c 1 -b 16 ref_audioF.wav trim 0.0 2.0 synth sine 880; \
    fi && \
    echo "Reference audio files ready:" && \
    ls -la ref_audio*.wav

# Verify all critical files are present
RUN echo "Verifying all required files..." && \
    ls -la main.py inferenceMSP.py ref_audio*.wav && \
    echo "File verification complete"

# Create non-root user for security (platform-agnostic)
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Create startup script that adapts to different platforms
RUN echo '#!/bin/bash\n\
echo "Starting Arabic MSP-TTS API..."\n\
echo "Platform: ${PLATFORM:-unknown}"\n\
echo "Port: ${PORT:-8000}"\n\
echo "Workers: ${WEB_CONCURRENCY:-1}"\n\
\n\
# Platform detection and adaptation\n\
if [ "$PLATFORM" = "heroku" ] || [ -n "$DYNO" ]; then\n\
    echo "Detected Heroku platform"\n\
    exec gunicorn main:app -w ${WEB_CONCURRENCY:-1} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120\n\
elif [ "$PLATFORM" = "railway" ] || [ -n "$RAILWAY_ENVIRONMENT" ]; then\n\
    echo "Detected Railway platform"\n\
    exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1\n\
elif [ "$PLATFORM" = "render" ] || [ -n "$RENDER" ]; then\n\
    echo "Detected Render platform"\n\
    exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 0\n\
elif [ "$PLATFORM" = "flyio" ] || [ -n "$FLY_APP_NAME" ]; then\n\
    echo "Detected Fly.io platform"\n\
    exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 5\n\
elif [ "$PLATFORM" = "gcp" ] || [ -n "$GOOGLE_CLOUD_PROJECT" ]; then\n\
    echo "Detected Google Cloud Platform"\n\
    exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 0\n\
else\n\
    echo "Using default configuration"\n\
    exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers ${WEB_CONCURRENCY:-1}\n\
fi' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port (standard across platforms)
EXPOSE 8000

# Use the adaptive startup script
CMD ["/app/start.sh"]