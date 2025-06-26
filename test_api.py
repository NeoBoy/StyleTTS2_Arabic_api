import requests
import time
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

# Test health endpoint
def test_health():
    """Test API health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    print(f"Health check passed: {response.json()}\n")

# Test device switching endpoint
def test_set_device(device: str):
    """Test switching between CPU and GPU"""
    print(f"Testing device switch to {device}...")
    payload = {"device": device}
    response = requests.post(f"{BASE_URL}/set_device/", json=payload)
    assert response.status_code == 200, f"Failed to switch to {device}: {response.status_code}"
    print(f"Device switched to {device}: {response.json()}\n")

# Test TTS generation for male and female voices
def test_tts_generation(text: str):
    """Test TTS generation for both male and female voices"""
    print(f"Testing TTS generation for text: {text}")

    # Male Voice
    print("Testing with male voice...")
    male_payload = {
        "text": text,
        "speaker_gender": "Male",
        "device": "cuda"  # Optional, you can also test with "cpu"
    }
    male_response = requests.post(f"{BASE_URL}/tts/", json=male_payload)
    if male_response.status_code == 200:
        with open("synthesized_audio_male.wav", "wb") as f:
            f.write(male_response.content)
        print("✅ Male voice TTS generated successfully and saved as 'synthesized_audio_male.wav'")
    else:
        print(f"❌ Error: {male_response.status_code}\n{male_response.text}")

    # Female Voice
    print("Testing with female voice...")
    female_payload = {
        "text": text,
        "speaker_gender": "Female",
        "device": "cuda"  # Optional, you can also test with "cpu"
    }
    female_response = requests.post(f"{BASE_URL}/tts/", json=female_payload)
    if female_response.status_code == 200:
        with open("synthesized_audio_female.wav", "wb") as f:
            f.write(female_response.content)
        print("✅ Female voice TTS generated successfully and saved as 'synthesized_audio_female.wav'")
    else:
        print(f"❌ Error: {female_response.status_code}\n{female_response.text}")
    print()

# Test switching device and generating TTS
def test_device_and_tts():
    """Test device switching and TTS generation"""
    text = "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ"

    # Switch to CPU
    test_set_device("cpu")

    # Test generating TTS on CPU
    test_tts_generation(text)

    # Switch to GPU (if available)
    if torch.cuda.is_available():
        test_set_device("cuda")
        # Test generating TTS on GPU
        test_tts_generation(text)

# Main test function
def main():
    """Run all tests"""
    print("🚀 Testing Arabic Multi-Speaker TTS API\n")

    # Test health check
    test_health()

    # Test device switching and TTS generation
    test_device_and_tts()

    print("🎉 Testing completed!")

if __name__ == "__main__":
    main()
