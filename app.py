import os, torch, yaml, time, phonemizer
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io.wavfile import write as write_wav
from typing import Optional
import tempfile
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
    language: str = "arabic"
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

def inferenceMSP_en(model, model_params, phonemes, sampler, device, diffusion_steps=5, embedding_scale=1, 
                    ref_audio='./ref_audio.wav', no_diff=False, alpha = 0.3, beta = 0.7):
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
        
        reference_vector = alpha * reference_vector + (1 - alpha)  * ref_s[:, :128]
        style_vector = beta * style_vector + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([reference_vector, style_vector], dim=-1)

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
async def generate_tts(
    text: str = Form(...),
    speaker_gender: Optional[str] = Form(default=None),
    language: str = Form(default="arabic"),
    device: Optional[str] = Form(default=None),
    reference_audio: Optional[UploadFile] = File(default=None)
):
    """
    Generate TTS audio with optional custom reference audio.
    
    - **text**: Input text in the specified language
    - **speaker_gender**: Voice gender selection (Male or Female)
    - **language**: Target language (arabic or english)
    - **device**: Optional device selection (cpu or cuda)
    - **reference_audio**: Optional custom reference audio file for voice cloning
    
    **Important**: When reference_audio is provided, the voice characteristics are determined entirely 
    by the reference audio file. The speaker_gender parameter is ignored in this case.
    
    If no reference_audio is provided, uses default voice files based on language and gender.
    """
    
    # Initialize temp_ref_path to None at the start to avoid UnboundLocalError
    temp_ref_path = None
    
    try:
        # Validate required parameters
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="'text' parameter is required and cannot be empty")

        if speaker_gender not in ["Male", "Female"]:
            raise HTTPException(status_code=400, detail="'speaker_gender' must be 'Male' or 'Female'")

        if language not in ["arabic", "english"]:
            raise HTTPException(status_code=400, detail="'language' must be 'arabic' or 'english'")
        
        # Validate language availability
        if language not in MODEL_CACHE or not MODEL_CACHE[language]:
            available_langs = [lang for lang in MODEL_CACHE if MODEL_CACHE[lang]]
            raise HTTPException(
                status_code=400, 
                detail=f"Language '{language}' not available. Available languages: {available_langs}"
            )
        
        # Determine device
        dev = device or ("cuda" if language in MODEL_CACHE and "cuda" in MODEL_CACHE[language] else "cpu")
        if dev not in MODEL_CACHE[language]:
            available_devices = list(MODEL_CACHE[language].keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Device '{dev}' not available for language '{language}'. Available devices: {available_devices}"
            )
        
        model, params = MODEL_CACHE[language][dev]
        ref_type = "default"
        
        # Handle reference audio selection
        if reference_audio:
            # User provided custom reference audio
            import mimetypes
            filename_lower = reference_audio.filename.lower()
            ct = reference_audio.content_type or ""
            guessed_ct, _ = mimetypes.guess_type(reference_audio.filename)
            allowed_ext = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
            
            is_ext_ok = filename_lower.endswith(allowed_ext)
            is_ct_ok = ct.startswith('audio/')
            is_guess_ok = (guessed_ct or '').startswith('audio/')
            
            if not (is_ct_ok or is_ext_ok or is_guess_ok):
                print(f"[VALIDATION] Rejecting file: name={reference_audio.filename} "
                      f"content_type='{ct}' guessed='{guessed_ct}' size_header={reference_audio.headers.get('content-length')}")
                raise HTTPException(status_code=400, detail="Reference file must be an audio file (wav/mp3/flac/ogg)")
            
            # Save uploaded file to temporary location
            temp_dir = tempfile.mkdtemp()
            temp_ref_path = os.path.join(temp_dir, f"custom_ref_{int(time.time())}.wav")
            
            content = await reference_audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded reference audio is empty")
            # Optional: size limit (e.g. 5 MB)
            if len(content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Reference audio exceeds 5MB limit")
            
            with open(temp_ref_path, "wb") as f:
                f.write(content)
    
            ref = temp_ref_path
            ref_type = "custom"
            print(f"🎤 CUSTOM REFERENCE DETECTED:")
            print(f"Using custom reference audio: {reference_audio.filename} "
                  f"(ct='{ct}' guessed='{guessed_ct}' bytes={len(content)})")            
            print(f"   File size: {len(content)} bytes")
            print(f"   Temp path: {temp_ref_path}")
            print(f"   File exists: {os.path.exists(temp_ref_path)}")
            
        else:
            if language == "arabic":
                if speaker_gender == "Male":
                    ref = "ref_audioM.wav"
                elif speaker_gender == "Female":
                    ref = "ref_audioF.wav"
                else:
                    raise HTTPException(status_code=400, detail="speaker_gender must be 'Male' or 'Female'")
            
            elif language == "english":
                if speaker_gender == "Male":
                    ref = "models/reference_audio/4077-13754-0000.wav"
                elif speaker_gender == "Female":
                    ref = "models/reference_audio/1221-135767-0014.wav"
                else:
                    raise HTTPException(status_code=400, detail="speaker_gender must be 'Male' or 'Female'")
            
            ref_type = "default"
            print(f"Using default reference audio: {ref}")
        
        # Verify reference audio file exists
        if not os.path.exists(ref):
            raise HTTPException(status_code=500, detail=f"Reference audio file not found: {ref}")
        
        # Language-specific configuration
        if language == "arabic":
            backend = phonemizer.backend.EspeakBackend(
                language="ar", preserve_punctuation=True, with_stress=True
            )
            no_diff = True
            diffusion_steps = 5
            inferenceFUN = inferenceMSP_fastapi
        
        elif language == "english":
            backend = phonemizer.backend.EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )
            no_diff = False
            alpha = 0.3
            beta = 0.7
            diffusion_steps = 5
            inferenceFUN = inferenceMSP_en
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        # Phonemize the text
        try:
            phonemes = backend.phonemize([text])[0]
            if not phonemes.strip():
                raise HTTPException(status_code=400, detail="Failed to phonemize text - text may contain unsupported characters")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Phonemization failed: {str(e)}")

        # Run inference
        start = time.time()
        try:
            if language == "english":
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
                    alpha=alpha,
                    beta=beta,
                )
            else:
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
            print(f"[{language}][{dev}][{ref_type}] Inference: {inference_time:.2f}s, no_diff={no_diff}")
            
        except Exception as e:
            print(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

        # Prepare audio response
        try:
            buf = BytesIO()
            write_wav(buf, 24000, wav)
            buf.seek(0)
            
            headers = {
                "X-Device": dev,
                "X-Language": language,
                "X-Speaker-Gender": speaker_gender if not reference_audio else "custom-reference",
                "X-Inference-Time": f"{inference_time:.2f}s",
                "X-Reference-Type": ref_type,
                "X-Reference-File": reference_audio.filename if reference_audio else "default",
                "Content-Disposition": f"attachment; filename=tts_{language}_{speaker_gender}_{ref_type}.wav"
            }

            # Add warning header when custom reference overrides gender
            if reference_audio and speaker_gender:
                headers["X-Warning"] = "speaker_gender parameter ignored when custom reference audio is provided"

            return StreamingResponse(buf, media_type="audio/wav", headers=headers)
        except Exception as e:
            print(f"Audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Log unexpected errors for debugging
        import traceback
        print(f"🚨 Unexpected error in TTS generation:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
        
    finally:
        # Clean up temporary file - now temp_ref_path is always defined
        if temp_ref_path and os.path.exists(temp_ref_path):
            try:
                os.remove(temp_ref_path)
                # Only remove directory if it's empty and was created by us
                temp_dir = os.path.dirname(temp_ref_path)
                if temp_dir.startswith('/tmp/tmp') and os.path.exists(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass  # Directory not empty, that's fine
                print(f"Cleaned up temporary reference file: {temp_ref_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary file: {e}")