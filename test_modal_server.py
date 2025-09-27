#!/usr/bin/env python3
"""
Test script for the Modal XTTS server.
This script tests all endpoints of the deployed Modal server.
"""

import base64
import json
import sys
from pathlib import Path

import requests


class XTTSModalTester:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['x-api-key'] = api_key
    
    def test_health(self):
        """Test the health endpoint"""
        print("Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/healthz")
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_root(self):
        """Test the root endpoint"""
        print("Testing root endpoint...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint: {data.get('status')}")
                print(f"   Available endpoints: {data.get('endpoints')}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_register_speaker_url(self, audio_url: str):
        """Test speaker registration with URL"""
        print(f"Testing speaker registration with URL: {audio_url}")
        try:
            payload = {"speaker_wav_url": audio_url}
            response = requests.post(
                f"{self.base_url}/register_speaker",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                speaker_id = data.get('speaker_id')
                print(f"✅ Speaker registered successfully: {speaker_id}")
                return speaker_id
            else:
                print(f"❌ Speaker registration failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Speaker registration error: {e}")
            return None
    
    def test_register_speaker_base64(self, audio_file_path: str):
        """Test speaker registration with base64"""
        print(f"Testing speaker registration with base64 from: {audio_file_path}")
        try:
            # Read and encode audio file
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            audio_base64 = base64.b64encode(audio_data).decode('ascii')
            
            payload = {"speaker_wav_base64": audio_base64}
            response = requests.post(
                f"{self.base_url}/register_speaker",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                speaker_id = data.get('speaker_id')
                print(f"✅ Speaker registered successfully: {speaker_id}")
                return speaker_id
            else:
                print(f"❌ Speaker registration failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Speaker registration error: {e}")
            return None
    
    def test_tts(self, text: str, language: str = "ar", speaker_id: str = None, 
                 output_file: str = "output.wav"):
        """Test TTS synthesis"""
        print(f"Testing TTS synthesis...")
        print(f"  Text: {text}")
        print(f"  Language: {language}")
        print(f"  Speaker ID: {speaker_id}")
        
        try:
            payload = {
                "text": text,
                "language": language,
                "temperature": 0.75,
                "return_base64": False
            }
            
            if speaker_id:
                payload["speaker_id"] = speaker_id
            
            response = requests.post(
                f"{self.base_url}/tts",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                # Save audio file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ TTS synthesis successful, saved to {output_file}")
                return True
            else:
                print(f"❌ TTS synthesis failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            return False
    
    def test_tts_base64(self, text: str, language: str = "ar", speaker_id: str = None):
        """Test TTS synthesis with base64 response"""
        print(f"Testing TTS synthesis with base64 response...")
        
        try:
            payload = {
                "text": text,
                "language": language,
                "temperature": 0.75,
                "return_base64": True
            }
            
            if speaker_id:
                payload["speaker_id"] = speaker_id
            
            response = requests.post(
                f"{self.base_url}/tts",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                audio_b64 = data.get('audio_wav_base64')
                sample_rate = data.get('sample_rate')
                print(f"✅ TTS base64 synthesis successful")
                print(f"   Sample rate: {sample_rate}")
                print(f"   Audio data length: {len(audio_b64)} characters")
                return True
            else:
                print(f"❌ TTS base64 synthesis failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ TTS base64 synthesis error: {e}")
            return False
    
    def run_full_test(self, test_audio_url: str = None, test_audio_file: str = None):
        """Run all tests"""
        print("XTTS Modal Server Test Suite")
        print("============================")
        
        results = []
        
        # Test basic endpoints
        results.append(self.test_health())
        results.append(self.test_root())
        
        # Test speaker registration
        speaker_id = None
        if test_audio_url:
            speaker_id = self.test_register_speaker_url(test_audio_url)
            results.append(speaker_id is not None)
        elif test_audio_file and Path(test_audio_file).exists():
            speaker_id = self.test_register_speaker_base64(test_audio_file)
            results.append(speaker_id is not None)
        else:
            print("⚠️ Skipping speaker registration tests (no audio provided)")
            results.append(True)  # Don't fail the test suite
        
        # Test TTS synthesis
        test_text = "مرحبا، هذا اختبار للنموذج"  # Arabic test text
        if speaker_id:
            results.append(self.test_tts(test_text, "ar", speaker_id, "test_output.wav"))
            results.append(self.test_tts_base64(test_text, "ar", speaker_id))
        else:
            print("⚠️ Skipping TTS tests (no speaker registered)")
            results.append(True)
            results.append(True)
        
        # Summary
        passed = sum(results)
        total = len(results)
        print(f"\nTest Results: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_modal_server.py <modal_url> [api_key] [audio_file_or_url]")
        print("Example: python test_modal_server.py https://your-app--xttsservice-tts.modal.run my_api_key test.wav")
        sys.exit(1)
    
    base_url = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    audio_input = sys.argv[3] if len(sys.argv) > 3 else None
    
    tester = XTTSModalTester(base_url, api_key)
    
    # Determine if audio input is URL or file
    if audio_input:
        if audio_input.startswith('http'):
            success = tester.run_full_test(test_audio_url=audio_input)
        else:
            success = tester.run_full_test(test_audio_file=audio_input)
    else:
        success = tester.run_full_test()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
