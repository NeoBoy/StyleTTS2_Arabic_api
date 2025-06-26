import os
import torch
import yaml
import time  # Import time module to measure inference time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from scipy.io.wavfile import write as write_wav  # Import write_wav from scipy.io to save audio
from inferenceMSP import inferenceMSP, load_models, load_model_weights  # Import inferenceMSP from the refactored script
import phonemizer

# Initialize FastAPI app
app = FastAPI()

# Pydantic model to validate input data
class TTSRequest(BaseModel):
    text: str
    speaker_gender: str
    embedding_scale: float = 1.0  # Default embedding scale
    device: str = None  # Allow device selection (either "cpu" or "cuda")

class DeviceRequest(BaseModel):
    device: str  # "cpu" or "cuda"

# Model directory path
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Default device configuration (initially using GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model once when FastAPI starts
model_cache = {}

# Function to load the model and parameters
def load_model_once():
    # Dynamically check which model files are available in the models directory
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    if not model_files:
        raise HTTPException(status_code=500, detail="No model files found in the models directory.")

    # Select the latest model based on file naming convention (you can change this logic)
    model_filename = sorted(model_files)[-1]  # Choose the latest model (you could change this logic)
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.isfile(model_path):
        raise HTTPException(status_code=500, detail=f"Model file {model_filename} not found.")
    
    # Ensure the correct path to the config file
    config_path = "/app/models/config.yml"  # Corrected path to the config.yml file
    if not os.path.isfile(config_path):
        raise HTTPException(status_code=500, detail="Configuration file not found.")
    
    # Load the YAML configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)  # Load the configuration
    
    # Load model architecture
    model, model_params = load_models(config, device)

    # Load model weights
    model = load_model_weights(model, model_path, device)

    # Cache the model
    model_cache['model'] = model
    model_cache['params'] = model_params
    
    # Display the model and config paths 
    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")

    # Perform model warm-up (dummy forward pass)
    if device == "cuda":
        # Dummy input to warm up the model
        dummy_input = torch.randn(1, 256).to(device)  # Example dummy input tensor
        phonemes = "dummy phoneme data"  # Dummy phonemes (you can adapt this if necessary)
        
        # Use the reference audio for warm-up (choose one based on the gender)
        ref_audio = "ref_audioM.wav"  # Can use ref_audioM.wav or ref_audioF.wav for warm-up
        sampler = None  # No diffusion, as we are warming up with style transfer only
        
        # Call inferenceMSP with dummy data to warm up
        try:
            print("Warming up the model with dummy data...")
            inferenceMSP(
                model,
                model_params,
                phonemes,
                sampler,  # Pass None for sampler as no diffusion is being used
                device=device,
                diffusion_steps=5,  # Fixed number of diffusion steps (not used)
                embedding_scale=1.0,
                ref_audio=ref_audio,  # Dummy reference audio for the warm-up
                no_diff=True  # Apply style transfer without diffusion
            )
            print("Model warmed up successfully.")
        except Exception as e:
            print(f"Error during model warm-up: {e}")

@app.on_event("startup")
async def startup_event():
    # Display available devices and set default device
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected. Defaulting to GPU.")
        else:
            print(f"Using a single GPU: {torch.cuda.get_device_name(0)}")
        # Set GPU as default if both CPU and GPU are available
        if torch.cuda.is_available():
            global device
            device = "cuda"
    else:
        print("Using CPU. No GPU available.")
    
    load_model_once()  # Load models at the start of the FastAPI app

@app.post("/set_device/")
async def set_device(request: DeviceRequest):
    """Endpoint to switch between GPU and CPU"""
    global device
    if request.device not in ['cpu', 'cuda']:
        raise HTTPException(status_code=400, detail="Invalid device. Use 'cpu' or 'cuda'.")
    
    device = request.device
    load_model_once()  # Reload the model with the new device
    return {"message": f"Device switched to {device}", "current_device": device}

@app.post("/tts/")  # Endpoint to generate TTS
async def generate_tts(request: TTSRequest):
    # Use the device passed in the request, or default to the chosen one at startup
    tts_device = request.device if request.device else device
    
    # Retrieve the cached model and parameters
    model = model_cache.get('model')
    model_params = model_cache.get('params')

    if not model or not model_params:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")
    
    # Phonemize the text using phonemizer (Arabic-specific setup)
    phonemizer_backend = phonemizer.backend.EspeakBackend(language='ar', preserve_punctuation=True, with_stress=True)
    phonemes = phonemizer_backend.phonemize([request.text])[0]

    # Select the correct reference audio based on speaker gender
    if request.speaker_gender == "Male":
        ref_audio = "ref_audioM.wav"  # Male reference audio in the working directory
    elif request.speaker_gender == "Female":
        ref_audio = "ref_audioF.wav"  # Female reference audio in the working directory
    else:
        raise HTTPException(status_code=400, detail="Invalid speaker gender. Use 'Male' or 'Female'.")

    # Start timing the inference
    start_time = time.time()

    # Generate the TTS output using inferenceMSP (no sampler, diffusion not used)
    wav = inferenceMSP(
        model,
        model_params,
        phonemes,
        sampler=None,  # No diffusion
        device=tts_device,
        diffusion_steps=5,  # Fixed number of diffusion steps (won't be used)
        embedding_scale=request.embedding_scale,
        ref_audio=ref_audio,  # Use reference audio based on gender
        no_diff=True  # Always apply style without diffusion
    )

    # End timing the inference
    end_time = time.time()
    inference_time = end_time - start_time  # Time in seconds

    # Log the inference time (printing it out, but not returning it in the response)
    print(f"Inference Time: {inference_time:.2f} seconds")

    # Save the generated audio to a file
    output_path = "/tmp/synthesized_audio.wav"
    write_wav(output_path, 24000, wav)  # Save at 24kHz sample rate

    # Return the audio file (no inference time in the response)
    return FileResponse(output_path)  # Return the synthesized audio file

@app.get("/")
def read_root():
    return {"message": "Arabic Multi-Speaker TTS API. Use POST /tts to generate speech."}
