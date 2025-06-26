import requests
import torch
from pathlib import Path

BASE_URL   = "http://localhost:8000"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_health():
    print("👉 Testing health endpoint…")
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    print("✅ Health OK:", resp.json(), "\n")

def test_tts(text: str):
    devices = [None, "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        label = dev or "default"
        for gender in ("Male", "Female"):
            print(f"👉 TTS | device={label:<7} | gender={gender}")
            payload = {"text": text, "speaker_gender": gender}
            if dev:
                payload["device"] = dev

            resp = requests.post(f"{BASE_URL}/tts/", json=payload)
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                fname = OUTPUT_DIR / f"tts_{gender.lower()}_{label}.wav"
                fname.write_bytes(resp.content)
                used = resp.headers.get("X-Device", label)
                print(f"   ✅ Saved → {fname} | device used: {used}\n")
            else:
                print(f"   ❌ Error body:\n{resp.text[:300]}…\n")

def main():
    print("\n🚀 Starting API tests\n")
    test_health()
    test_tts("السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ")
    print("🎉 Tests completed. Check the `outputs/` folder for WAV files.\n")

if __name__ == "__main__":
    main()
