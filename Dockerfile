# Use the official PyTorch image with CUDA 2.5 and Ubuntu 24.04 support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and other necessary libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    espeak-ng \
    libsndfile1 \
    build-essential \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install phonemizer librosa scipy pydantic fastapi uvicorn

# Clone the repository directly into the current working directory
RUN git clone https://github.com/NeoBoy/StyleTTS2_Arabic.git .

# Pull the latest changes from the cloned repository
RUN git pull origin main  # Pull the latest changes from the 'main' branch

# Install the repository's Python dependencies
RUN pip install -r /app/requirements.txt

# Download the model files from Hugging Face
WORKDIR /app/models
RUN wget -O model24.pth https://huggingface.co/blits-ai/style_tts2_finetune_audiobook4GPU/resolve/main/epoch_00024.pth
RUN wget -O config.yml https://huggingface.co/fadi77/StyleTTS2-LibriTTS-arabic/resolve/main/config.yml


# Reset the working directory back to /app to run the FastAPI app
WORKDIR /app

# Ensure the 'app.py' exists in the root directory
COPY app.py /app/app.py

# Expose port 8000 for FastAPI to run on
EXPOSE 8000

# Set the command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
