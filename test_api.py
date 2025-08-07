import requests
import torch
import time
from pathlib import Path

BASE_URL   = "http://localhost:8000"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_health():
    print("👉 Testing health endpoint...")
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    print("✅ Health OK:")
    print(f"   Available languages: {data.get('available_languages', [])}")
    print(f"   Total model instances: {data.get('total_model_instances', 0)}")
    print(f"   Device info: {data.get('device_info', {})}")
    print()

def test_health_endpoint():
    print("👉 Testing /health endpoint...")
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Health endpoint failed: {resp.status_code}"
    data = resp.json()
    print("✅ Health endpoint OK:")
    print(f"   Status: {data.get('status')}")
    print(f"   Available languages: {data.get('available_languages', [])}")
    print(f"   CUDA available: {data.get('cuda_available')}")
    print(f"   Devices: {data.get('devices', [])}")
    print()

def test_tts_arabic(text: str):
    print("🔤 Testing Arabic TTS...")
    devices = [None, "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        label = dev or "default"
        for gender in ("Male", "Female"):
            print(f"👉 Arabic TTS | device={label:<7} | gender={gender}")
            payload = {
                "text": text, 
                "speaker_gender": gender,
                "language": "arabic"
            }
            if dev:
                payload["device"] = dev

            start_time = time.time()
            resp = requests.post(f"{BASE_URL}/tts/", json=payload)
            request_time = time.time() - start_time
            
            print(f"   Status: {resp.status_code} | Request time: {request_time:.2f}s")
            if resp.status_code == 200:
                fname = OUTPUT_DIR / f"arabic_tts_{gender.lower()}_{label}.wav"
                fname.write_bytes(resp.content)
                used_device = resp.headers.get("X-Device", label)
                inference_time = resp.headers.get("X-Inference-Time", "N/A")
                no_diff = resp.headers.get("X-No-Diff", "N/A")
                print(f"   ✅ Saved → {fname}")
                print(f"   📊 Device used: {used_device} | Inference: {inference_time} | No-diff: {no_diff}\n")
            else:
                print(f"   ❌ Error body:\n{resp.text[:300]}...\n")

def test_tts_english(text: str):
    print("🔤 Testing English TTS...")
    devices = [None, "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        label = dev or "default"
        for gender in ("Male", "Female"):
            print(f"👉 English TTS | device={label:<7} | gender={gender}")
            payload = {
                "text": text, 
                "speaker_gender": gender,
                "language": "english"
            }
            if dev:
                payload["device"] = dev

            start_time = time.time()
            resp = requests.post(f"{BASE_URL}/tts/", json=payload)
            request_time = time.time() - start_time
            
            print(f"   Status: {resp.status_code} | Request time: {request_time:.2f}s")
            if resp.status_code == 200:
                fname = OUTPUT_DIR / f"english_tts_{gender.lower()}_{label}.wav"
                fname.write_bytes(resp.content)
                used_device = resp.headers.get("X-Device", label)
                inference_time = resp.headers.get("X-Inference-Time", "N/A")
                no_diff = resp.headers.get("X-No-Diff", "N/A")
                print(f"   ✅ Saved → {fname}")
                print(f"   📊 Device used: {used_device} | Inference: {inference_time} | No-diff: {no_diff}\n")
            else:
                print(f"   ❌ Error body:\n{resp.text[:300]}...\n")

def test_error_cases():
    print("🚨 Testing error cases...")
    
    # Test unsupported language
    print("👉 Testing unsupported language...")
    payload = {"text": "Hello", "speaker_gender": "Male", "language": "french"}
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    print(f"   Status: {resp.status_code} (expected 400)")
    if resp.status_code == 400:
        print("   ✅ Correctly rejected unsupported language\n")
    else:
        print(f"   ❌ Unexpected response: {resp.text[:200]}...\n")
    
    # Test invalid gender
    print("👉 Testing invalid gender...")
    payload = {"text": "مرحبا", "speaker_gender": "Robot", "language": "arabic"}
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    print(f"   Status: {resp.status_code} (expected 400)")
    if resp.status_code == 400:
        print("   ✅ Correctly rejected invalid gender\n")
    else:
        print(f"   ❌ Unexpected response: {resp.text[:200]}...\n")
    
    # Test empty text
    print("👉 Testing empty text...")
    payload = {"text": "", "speaker_gender": "Male", "language": "arabic"}
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    print(f"   Status: {resp.status_code} (expected 400)")
    if resp.status_code == 400:
        print("   ✅ Correctly rejected empty text\n")
    else:
        print(f"   ❌ Unexpected response: {resp.text[:200]}...\n")
    
    # Test unsupported device
    print("👉 Testing unsupported device...")
    payload = {"text": "Hello", "speaker_gender": "Male", "language": "english", "device": "tpu"}
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    print(f"   Status: {resp.status_code} (expected 400)")
    if resp.status_code == 400:
        print("   ✅ Correctly rejected unsupported device\n")
    else:
        print(f"   ❌ Unexpected response: {resp.text[:200]}...\n")
    
    # Test special characters
    print("👉 Testing special characters...")
    payload = {"text": "Hello! How are you? 123 @#$%", "speaker_gender": "Female", "language": "english"}
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        fname = OUTPUT_DIR / "english_special_chars.wav"
        fname.write_bytes(resp.content)
        print(f"   ✅ Special characters handled → {fname}\n")
    else:
        print(f"   ❌ Failed with special characters: {resp.text[:200]}...\n")

def test_performance():
    print("⚡ Testing performance...")
    
    # Test concurrent requests
    import threading
    import queue
    
    def make_request(q, text, lang, gender, device=None):
        payload = {"text": text, "speaker_gender": gender, "language": lang}
        if device:
            payload["device"] = device
        
        start = time.time()
        resp = requests.post(f"{BASE_URL}/tts/", json=payload)
        elapsed = time.time() - start
        q.put((resp.status_code, elapsed, resp.headers.get("X-Device", "unknown")))
    
    # Test 3 concurrent requests
    q = queue.Queue()
    threads = []
    
    requests_data = [
        ("مرحبا بكم", "arabic", "Male"),
        ("Hello world", "english", "Female"),
        ("كيف حالكم", "arabic", "Female")
    ]
    
    print("👉 Testing 3 concurrent requests...")
    start_time = time.time()
    
    for text, lang, gender in requests_data:
        t = threading.Thread(target=make_request, args=(q, text, lang, gender))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    results = []
    while not q.empty():
        results.append(q.get())
    
    print(f"   Total time for 3 concurrent requests: {total_time:.2f}s")
    for i, (status, elapsed, device) in enumerate(results):
        print(f"   Request {i+1}: {status} | {elapsed:.2f}s | device: {device}")
    print()

def test_large_text():
    print("📄 Testing large text...")
    
    # Large Arabic text
    large_arabic = "مرحبا بكم في خدمة التحويل من النص إلى الكلام. " * 10
    print(f"👉 Testing large Arabic text ({len(large_arabic)} chars)...")
    payload = {"text": large_arabic, "speaker_gender": "Male", "language": "arabic"}
    
    start = time.time()
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    elapsed = time.time() - start
    
    print(f"   Status: {resp.status_code} | Time: {elapsed:.2f}s")
    if resp.status_code == 200:
        fname = OUTPUT_DIR / "arabic_large_text.wav"
        fname.write_bytes(resp.content)
        print(f"   ✅ Large Arabic text handled → {fname}\n")
    else:
        print(f"   ❌ Failed: {resp.text[:200]}...\n")
    
    # Large English text
    large_english = "Welcome to our text-to-speech service. This is a test of a longer sentence. " * 8
    print(f"👉 Testing large English text ({len(large_english)} chars)...")
    payload = {"text": large_english, "speaker_gender": "Female", "language": "english"}
    
    start = time.time()
    resp = requests.post(f"{BASE_URL}/tts/", json=payload)
    elapsed = time.time() - start
    
    print(f"   Status: {resp.status_code} | Time: {elapsed:.2f}s")
    if resp.status_code == 200:
        fname = OUTPUT_DIR / "english_large_text.wav"
        fname.write_bytes(resp.content)
        print(f"   ✅ Large English text handled → {fname}\n")
    else:
        print(f"   ❌ Failed: {resp.text[:200]}...\n")

def main():
    print("🚀 Starting comprehensive API tests\n")
    print("=" * 60)
    
    try:
        # Basic health checks
        test_health()
        test_health_endpoint()
        
        # Language-specific tests
        test_tts_arabic("السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ")
        test_tts_english("Hello, welcome to our text-to-speech service")
        
        # Error handling tests
        test_error_cases()
        
        # Performance tests
        test_performance()
        
        # Large text tests
        test_large_text()
        
        print("=" * 60)
        print("🎉 All tests completed successfully!")
        print(f"📁 Check the '{OUTPUT_DIR}' folder for generated WAV files.")
        
        # List generated files
        wav_files = list(OUTPUT_DIR.glob("*.wav"))
        if wav_files:
            print(f"\n📊 Generated {len(wav_files)} audio files:")
            for f in sorted(wav_files):
                size_kb = f.stat().st_size / 1024
                print(f"   • {f.name} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

if __name__ == "__main__":
    main()