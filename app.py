import os, torch, yaml, time, phonemizer
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io.wavfile import write as write_wav
import librosa
import nltk
nltk.download('punkt_tab', quiet=True)  # Download punkt tokenizer

from nltk.tokenize import word_tokenize
from inferenceMSP import load_models, load_model_weights, inferenceMSP_fastapi, preprocess
from models import *
from utils import *

app = FastAPI()
MODEL_DIR = "./models"
ARABIC_CONFIG_PATH = os.path.join(MODEL_DIR, "config_ar.yml")
ENGLISH_CONFIG_PATH = os.path.join(MODEL_DIR, "config_en.yml")
ARABIC_MODEL_PATH = os.path.join(MODEL_DIR, "model_ar.pth")
ENGLISH_MODEL_PATH = os.path.join(MODEL_DIR, "model_en.pth")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MODEL_CACHE = {}  # will hold { "arabic": {"cpu": (model, params), "cuda": ...}, "english": {...} }

class TTSRequest(BaseModel):
    text: str
    speaker_gender: str
    language: str = "arabic"  # new parameter with default
    device: str = None

def fix_yaml_file(file_path):
    """Fix YAML file by replacing tabs with spaces"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace tabs with 2 spaces
        fixed_content = content.replace('\t', '  ')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✓ Fixed YAML formatting in {file_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to fix YAML file {file_path}: {e}")
        return False

def load_yaml_config(config_path):
    """Load YAML config with error handling and auto-fix"""
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.scanner.ScannerError as e:
        print(f"✗ YAML syntax error in {config_path}: {e}")
        print("Attempting to fix YAML formatting...")
        
        if fix_yaml_file(config_path):
            try:
                with open(config_path, "r", encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e2:
                print(f"✗ Still failed to load after fix: {e2}")
                return None
        return None
    except Exception as e:
        print(f"✗ Failed to load {config_path}: {e}")
        return None

def load_models_en(config, device):
    """Load all required models for inference."""
    
    # Load pretrained models
    text_aligner = load_ASR_models("Utils/ASR/epoch_00080.pth", "Utils/ASR/config.yml")

    pitch_extractor = load_F0_models("Utils/JDC/bst.t7")
    
    # Load BERT model
    from Utils.PLBERT_English.util import load_plbert_old
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert_old(BERT_path)

    # Build the main model
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    # Move models to device and set to eval mode
    for key in model:
        model[key].eval()
        model[key].to(device)
    
    return model, model_params

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

def compute_style_en(path, device, model):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


from text_utils import TextCleaner
textclenaer = TextCleaner()

def inferenceMSP_en(model, model_params, phonemes, sampler, device, diffusion_steps=5, 
                embedding_scale=1, ref_audio='./ref_audio.wav', no_diff=False):
    """Generate speech from phonemized text and speaker style."""
    
    sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
    )
    
    # Tokenize input phonemes
    ps = word_tokenize(phonemes)
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        
        # Style generation through diffusion
        if ref_audio is None:
            s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
            ).squeeze(1)
        elif no_diff:
            s_pred = compute_style_en(ref_audio, device, model)
        else:
            ref_s = compute_style_en(ref_audio, device, model)
            s_pred = sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps
            ).squeeze(1)

        # Split style vector into style and reference components
        style_vector = s_pred[:, 128:]
        reference_vector = s_pred[:, :128]

        # Duration prediction
        duration_encoding = model.predictor.text_encoder(d_en, style_vector, input_lengths, text_mask)
        lstm_output, _ = model.predictor.lstm(duration_encoding)
        duration_logits = model.predictor.duration_proj(lstm_output)
        
        # Process durations
        duration_probs = torch.sigmoid(duration_logits).sum(axis=-1)
        predicted_durations = torch.round(duration_probs.squeeze()).clamp(min=1)
        
        # Create alignment target
        alignment_target = torch.zeros(input_lengths, int(predicted_durations.sum().data))
        current_frame = 0
        for i in range(alignment_target.size(0)):
            dur_i = int(predicted_durations[i].data)
            alignment_target[i, current_frame:current_frame + dur_i] = 1
            current_frame += dur_i
        
        # Encode prosody
        prosody_encoding = (duration_encoding.transpose(-1, -2) @ alignment_target.unsqueeze(0).to(device))
        
        # Handle HifiGAN decoder specifics
        if model_params.decoder.type == "hifigan":
            shifted_encoding = torch.zeros_like(prosody_encoding)
            shifted_encoding[:, :, 0] = prosody_encoding[:, :, 0]
            shifted_encoding[:, :, 1:] = prosody_encoding[:, :, 0:-1]
            prosody_encoding = shifted_encoding
        
        # Predict F0 and noise
        f0_prediction, noise_prediction = model.predictor.F0Ntrain(prosody_encoding, style_vector)
        
        # Prepare ASR features
        asr_features = (t_en @ alignment_target.unsqueeze(0).to(device))
        
        # Handle HifiGAN decoder specifics for ASR features
        if model_params.decoder.type == "hifigan":
            shifted_asr = torch.zeros_like(asr_features)
            shifted_asr[:, :, 0] = asr_features[:, :, 0]
            shifted_asr[:, :, 1:] = asr_features[:, :, 0:-1]
            asr_features = shifted_asr
        
        # Generate audio
        audio_output = model.decoder(
            asr_features,
            f0_prediction, 
            noise_prediction, 
            reference_vector.squeeze().unsqueeze(0)
        )
    
    # Remove artifacts at the end of the audio
    return audio_output.squeeze().cpu().numpy()[..., :-50]

def init_models():
    MODEL_CACHE["arabic"] = {}
    MODEL_CACHE["english"] = {}
    
    # Load Arabic models
    if os.path.isfile(ARABIC_CONFIG_PATH) and os.path.isfile(ARABIC_MODEL_PATH):
        print(f"Loading Arabic model from {ARABIC_MODEL_PATH}")
        
        arabic_cfg = load_yaml_config(ARABIC_CONFIG_PATH)
        if arabic_cfg is None:
            print("✗ Failed to load Arabic config, skipping Arabic model")
        else:
            for dev in DEVICES:
                try:
                    model, params = load_models(arabic_cfg, dev)
                    for k in model:
                        model[k] = model[k].to(dev)
                    model = load_model_weights(model, ARABIC_MODEL_PATH, dev)
                    for k in model:
                        model[k] = model[k].to(dev)
                    MODEL_CACHE["arabic"][dev] = (model, params)
                    print(f"✓ Loaded Arabic model on {dev}")
                except Exception as e:
                    print(f"✗ Failed to load Arabic model on {dev}: {e}")
    else:
        print("✗ Arabic model files not found")
        print(f"  Config: {ARABIC_CONFIG_PATH} exists: {os.path.isfile(ARABIC_CONFIG_PATH)}")
        print(f"  Model: {ARABIC_MODEL_PATH} exists: {os.path.isfile(ARABIC_MODEL_PATH)}")
    
    # Load English models
    if os.path.isfile(ENGLISH_CONFIG_PATH) and os.path.isfile(ENGLISH_MODEL_PATH):
        print(f"Loading English model from {ENGLISH_MODEL_PATH}")
        
        english_cfg = load_yaml_config(ENGLISH_CONFIG_PATH)
        if english_cfg is None:
            print("✗ Failed to load English config, skipping English model")
        else:
            for dev in DEVICES:
                try:
                    model, params = load_models_en(english_cfg, dev)
                    for k in model:
                        model[k] = model[k].to(dev)
                    model = load_model_weights(model, ENGLISH_MODEL_PATH, dev)
                    for k in model:
                        model[k] = model[k].to(dev)
                    MODEL_CACHE["english"][dev] = (model, params)
                    print(f"✓ Loaded English model on {dev}")
                except Exception as e:
                    print(f"✗ Failed to load English model on {dev}: {e}")
    else:
        print("✗ English model files not found")
        print(f"  Config: {ENGLISH_CONFIG_PATH} exists: {os.path.isfile(ENGLISH_CONFIG_PATH)}")
        print(f"  Model: {ENGLISH_MODEL_PATH} exists: {os.path.isfile(ENGLISH_MODEL_PATH)}")

@app.on_event("startup")
async def on_startup():
    # 1) Load models
    print("=" * 50)
    print("Initializing TTS Models...")
    print("=" * 50)
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"✗ Model directory {MODEL_DIR} does not exist")
        return
    
    # List files in models directory for debugging
    print(f"Files in {MODEL_DIR}:")
    for file in os.listdir(MODEL_DIR):
        file_path = os.path.join(MODEL_DIR, file)
        size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
        print(f"  {file} ({size} bytes)")
    
    init_models()
    
    print("\nModels ready:")
    for lang in MODEL_CACHE:
        if MODEL_CACHE[lang]:
            print(f"  {lang}: {', '.join(MODEL_CACHE[lang].keys())}")
        else:
            print(f"  {lang}: not available")

    # 2) Warm up CUDA for both languages
    if any(MODEL_CACHE[lang].get("cuda") for lang in MODEL_CACHE):
        print("\nWarming up CUDA models...")
        try:
            # Warm up Arabic model
            if MODEL_CACHE["arabic"].get("cuda"):
                dev = "cuda"
                model, params = MODEL_CACHE["arabic"][dev]
                backend = phonemizer.backend.EspeakBackend(
                    language="ar", preserve_punctuation=True, with_stress=True
                )
                dummy_text = "سلام"
                dummy_phonemes = backend.phonemize([dummy_text])[0]
                ref = "ref_audioM.wav"  # Arabic reference audio
                _ = inferenceMSP_fastapi(
                    model=model,
                    model_params=params,
                    phonemes=dummy_phonemes,
                    sampler=None,
                    device=dev,
                    diffusion_steps=5,
                    embedding_scale=1.0,
                    ref_audio=ref,
                    no_diff=True,  # Arabic uses no_diff=True
                )
                print("✓ Arabic CUDA warmup completed (no_diff=True)")
            
            # Warm up English model
            if MODEL_CACHE["english"].get("cuda"):
                dev = "cuda"
                model, params = MODEL_CACHE["english"][dev]
                backend = phonemizer.backend.EspeakBackend(
                    language="en-us", preserve_punctuation=True, with_stress=True
                )
                dummy_text = "Hello"
                dummy_phonemes = backend.phonemize([dummy_text])[0]
                ref = "models/reference_audio/4077-13754-0000.wav"  # English reference audio
                _ = inferenceMSP_en(
                    model=model,
                    model_params=params,
                    phonemes=dummy_phonemes,
                    sampler=None,
                    device=dev,
                    diffusion_steps=5,
                    embedding_scale=1.0,
                    ref_audio=ref,
                    no_diff=False,  # English uses no_diff=False
                )
                print("✓ English CUDA warmup completed (no_diff=False)")
                
        except Exception as e:
            print(f"✗ CUDA warmup failed: {e}")
    
    print("=" * 50)
    print("TTS API Ready!")
    print("=" * 50)

@app.get("/")
def read_root():
    available_langs = [lang for lang in MODEL_CACHE if MODEL_CACHE[lang]]
    total_devices = sum(len(MODEL_CACHE[lang]) for lang in available_langs)
    
    return {
        "message": "Multi-language TTS API",
        "description": "Arabic and English Text-to-Speech service",
        "available_languages": available_langs,
        "supported_genders": ["Male", "Female"],
        "total_model_instances": total_devices,
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

@app.get("/health")
def health_check():
    available_langs = [lang for lang in MODEL_CACHE if MODEL_CACHE[lang]]
    return {
        "status": "healthy" if available_langs else "unhealthy",
        "available_languages": available_langs,
        "cuda_available": torch.cuda.is_available(),
        "devices": DEVICES
    }

@app.post("/tts/")
def generate_tts(req: TTSRequest):
    # Validate language
    if req.language not in MODEL_CACHE or not MODEL_CACHE[req.language]:
        available_langs = [lang for lang in MODEL_CACHE if MODEL_CACHE[lang]]
        raise HTTPException(
            status_code=400, 
            detail=f"Language '{req.language}' not available. Available languages: {available_langs}"
        )
    
    # Determine device
    dev = req.device or ("cuda" if req.language in MODEL_CACHE and "cuda" in MODEL_CACHE[req.language] else "cpu")
    if dev not in MODEL_CACHE[req.language]:
        available_devices = list(MODEL_CACHE[req.language].keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Device '{dev}' not available for language '{req.language}'. Available devices: {available_devices}"
        )
    
    model, params = MODEL_CACHE[req.language][dev]

    # Language-specific configuration
    if req.language == "arabic":
        # Arabic language settings
        backend = phonemizer.backend.EspeakBackend(
            language="ar", preserve_punctuation=True, with_stress=True
        )
        
        # Reference audio selection for Arabic
        if req.speaker_gender == "Male":
            ref = "ref_audioM.wav"  # Arabic male reference
        elif req.speaker_gender == "Female":
            ref = "ref_audioF.wav"  # Arabic female reference
        else:
            raise HTTPException(status_code=400, detail="speaker_gender must be 'Male' or 'Female'")
        
        no_diff = True  # Arabic model uses no diffusion
        diffusion_steps = 5  # Steps for Arabic
        inferenceFUN = inferenceMSP_fastapi  # Use Arabic inference function

    elif req.language == "english":
        # English language settings
        backend = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        
        # Reference audio selection for English
        if req.speaker_gender == "Male":
            ref = "models/reference_audio/4077-13754-0000.wav"  # English male reference
        elif req.speaker_gender == "Female":
            ref = "models/reference_audio/1221-135767-0014.wav"  # English female reference
        else:
            raise HTTPException(status_code=400, detail="speaker_gender must be 'Male' or 'Female'")
        
        no_diff = False  # English model also uses no diffusion
        diffusion_steps = 5  # Same steps as Arabic
        inferenceFUN = inferenceMSP_en  # Use English inference function
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")

    # Validate text input
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Phonemize the text
    try:
        phonemes = backend.phonemize([req.text])[0]
        if not phonemes.strip():
            raise HTTPException(status_code=400, detail="Failed to phonemize text - text may contain unsupported characters")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Phonemization failed: {str(e)}")

    # Run inference
    start = time.time()
    try:
        wav = inferenceFUN(
            model=model,
            model_params=params,
            phonemes=phonemes,
            sampler=None,
            device=dev,
            diffusion_steps=diffusion_steps,
            embedding_scale=1.0,
            ref_audio=ref,
            no_diff=no_diff,
        )
        inference_time = time.time() - start
        print(f"[{req.language}][{dev}] Inference: {inference_time:.2f}s, no_diff={no_diff}, steps={diffusion_steps}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

    # Prepare audio response
    try:
        buf = BytesIO()
        write_wav(buf, 24000, wav)
        buf.seek(0)
        
        return StreamingResponse(
            buf, 
            media_type="audio/wav",
            headers={
                "X-Device": dev,
                "X-Language": req.language,
                "X-Speaker-Gender": req.speaker_gender,
                "X-Inference-Time": f"{inference_time:.2f}s",
                "X-No-Diff": str(no_diff),
                "Content-Disposition": f"attachment; filename=tts_{req.language}_{req.speaker_gender}.wav"
            }
        )
    except Exception as e:
        print(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")