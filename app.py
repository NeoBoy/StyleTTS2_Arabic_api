import os, torch, yaml, time, phonemizer
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io.wavfile import write as write_wav
from inferenceMSP import load_models, load_model_weights, inferenceMSP_fastapi

app = FastAPI()
MODEL_DIR = "./models"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yml")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MODEL_CACHE = {}  # will hold { "cpu": (model, params), "cuda": ... }

class TTSRequest(BaseModel):
    text: str
    speaker_gender: str
    device: str = None

def init_models():
    if not os.path.isfile(CONFIG_PATH):
        raise RuntimeError(f"config.yml not found in {MODEL_DIR}")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    ckpts = sorted([fn for fn in os.listdir(MODEL_DIR) if fn.endswith(".pth")])
    if not ckpts:
        raise RuntimeError("No .pth checkpoint in models/")
    latest = os.path.join(MODEL_DIR, ckpts[-1])

    for dev in DEVICES:
        model, params = load_models(cfg, dev)
        for k in model:
            model[k] = model[k].to(dev)
        model = load_model_weights(model, latest, dev)
        for k in model:
            model[k] = model[k].to(dev)
        MODEL_CACHE[dev] = (model, params)
        print(f"Loaded {latest} onto {dev}")

@app.on_event("startup")
async def on_startup():
    # 1) Load models
    init_models()
    print("Models ready on devices:", ", ".join(MODEL_CACHE.keys()))

    # 2) Warm up CUDA (dummy inference) to JIT kernels & allocate memory
    if "cuda" in MODEL_CACHE:
        try:
            dev = "cuda"
            model, params = MODEL_CACHE[dev]
            # use the same phonemizer backend as in /tts endpoint
            backend = phonemizer.backend.EspeakBackend(
                language="ar", preserve_punctuation=True, with_stress=True
            )
            dummy_text = "سلام"
            dummy_phonemes = backend.phonemize([dummy_text])[0]
            # choose a reference audio (Male by default)
            ref = "ref_audioM.wav"
            # run a very short dummy inference
            _ = inferenceMSP_fastapi(
                model=model,
                model_params=params,
                phonemes=dummy_phonemes,
                sampler=None,
                device=dev,
                diffusion_steps=5,
                embedding_scale=1.0,
                ref_audio=ref,
                no_diff=True,
            )
            print("[warmup][cuda] dummy inference completed")
        except Exception as e:
            print(f"[warmup][cuda] dummy inference failed: {e}")

@app.get("/")
def read_root():
    return {"message": "Arabic Multi-Speaker TTS API. Use POST /tts to generate speech."}

@app.post("/tts/")
def generate_tts(req: TTSRequest):
    dev = req.device or ("cuda" if "cuda" in MODEL_CACHE else "cpu")
    if dev not in MODEL_CACHE:
        raise HTTPException(400, f"Device '{dev}' not available.")
    model, params = MODEL_CACHE[dev]

    # phonemize
    backend = phonemizer.backend.EspeakBackend(
        language="ar", preserve_punctuation=True, with_stress=True
    )
    phonemes = backend.phonemize([req.text])[0]

    # ref audio pick
    if req.speaker_gender == "Male":
        ref = "ref_audioM.wav"
    elif req.speaker_gender == "Female":
        ref = "ref_audioF.wav"
    else:
        raise HTTPException(400, "speaker_gender must be 'Male' or 'Female'")

    # inference
    start = time.time()
    wav = inferenceMSP_fastapi(
        model=model,
        model_params=params,
        phonemes=phonemes,
        sampler=None,
        device=dev,
        diffusion_steps=5,
        embedding_scale=1.0,
        ref_audio=ref,
        no_diff=True,
    )
    print(f"[{dev}] Inference: {time.time()-start:.2f}s")

    # stream back
    buf = BytesIO()
    write_wav(buf, 24000, wav)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav", headers={"X-Device": dev})
