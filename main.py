from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, glob, uuid, subprocess
from pathlib import Path

app = FastAPI(title="Arabic MSP-TTS API")

# Define directories and file paths
MODEL_DIR = "./models"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yml")
if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config.yml in {MODEL_DIR}")

# Automatically locate available .pth model files
model_paths = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
if not model_paths:
    raise FileNotFoundError(f"No .pth files found under {MODEL_DIR}")

model_choices = [os.path.basename(p) for p in model_paths]

# Ensure the required reference audio files exist in the container root
REF_AUDIO_M = "ref_audioM.wav"
REF_AUDIO_F = "ref_audioF.wav"
if not os.path.isfile(REF_AUDIO_M):
    raise FileNotFoundError(f"Missing {REF_AUDIO_M}")
if not os.path.isfile(REF_AUDIO_F):
    raise FileNotFoundError(f"Missing {REF_AUDIO_F}")

# Define valid generation modes
valid_gen_modes = [
    "Compute style only (no diffusion)",
    "Diffusion conditioned on style"
]

def run_text_to_speech(text: str, selected_model: str, speaker_gender: str, generation_mode: str) -> Path:
    if not text.strip():
        raise ValueError("Please enter some text to synthesize.")
    if selected_model not in model_choices:
        raise ValueError("Please select a valid TTS model.")
    if speaker_gender not in ("Male", "Female"):
        raise ValueError("Please choose either Male or Female for speaker gender.")
    
    # Determine flags and reference audio based on the generation mode
    ref_flag = []
    no_diff_flag = []
    if generation_mode == "Compute style only (no diffusion)":
        ref_file = REF_AUDIO_M if speaker_gender == "Male" else REF_AUDIO_F
        ref_flag = ["--ref_speaker_audio", ref_file]
        no_diff_flag = ["--no_diff"]
    elif generation_mode == "Diffusion conditioned on style":
        ref_file = REF_AUDIO_M if speaker_gender == "Male" else REF_AUDIO_F
        ref_flag = ["--ref_speaker_audio", ref_file]
    else:
        raise ValueError("Invalid generation mode selected.")
    
    # Create a unique output filename
    out_fname = f"./synth_{uuid.uuid4().hex}.wav"
    
    # Assemble the command to run inference
    cmd = [
        "python", "inferenceMSP.py",
        "--config", CONFIG_PATH,
        "--model", os.path.join(MODEL_DIR, selected_model),
        "--text", text,
        "--output", out_fname
    ] + ref_flag + no_diff_flag
    
    # Optional: log the command for debugging purposes
    print("Running command:", " ".join(cmd))
    
    # Execute the inference command
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if proc.returncode != 0:
        error_message = proc.stderr.strip() or proc.stdout.strip() or f"Exit code {proc.returncode}"
        raise RuntimeError(f"Inference failed: {error_message}")
    
    if not os.path.isfile(out_fname):
        raise RuntimeError(f"Output file not found: {out_fname}")
    
    return Path(out_fname)

# Define the expected payload structure
class SynthesisRequest(BaseModel):
    text: str
    selected_model: str
    speaker_gender: str
    generation_mode: str

# The main endpoint that handles TTS synthesis requests
@app.post("/synthesize", response_description="Synthesized audio")
def synthesize(request: SynthesisRequest):
    try:
        output_path = run_text_to_speech(
            request.text,
            request.selected_model,
            request.speaker_gender,
            request.generation_mode
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Return the synthesized audio file
    return FileResponse(path=str(output_path), filename=output_path.name, media_type="audio/wav")

# An additional helper endpoint to list available models if needed
@app.get("/models", response_description="Available TTS models")
def list_models():
    return {"models": model_choices}
