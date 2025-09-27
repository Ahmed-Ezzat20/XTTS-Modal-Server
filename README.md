_# XTTS Modal Server

A serverless deployment of XTTS (Coqui TTS) v2 on Modal Labs platform with public API endpoints for text-to-speech synthesis.

## 🎉 **LIVE DEPLOYMENT STATUS: FULLY OPERATIONAL**

**Working Endpoints (Deployed & Tested):**
- **Health Check**: https://ahmedezzat0247--xtts-server-fixed-xttsservice-healthz.modal.run
- **Service Info**: https://ahmedezzat0247--xtts-server-fixed-xttsservice-root.modal.run  
- **Register Speaker**: https://ahmedezzat0247--xtts-server-fixed-xttsservice-register-speaker.modal.run
- **Text-to-Speech**: https://ahmedezzat0247--xtts-server-fixed-xttsservice-tts.modal.run

✅ **Model**: Kuwaiti XTTS Latest (5.6GB) - Successfully loaded  
✅ **Languages**: Arabic & English synthesis confirmed  
✅ **Testing**: All endpoints verified and working  
✅ **Performance**: ~2-12 seconds per request (warm containers)

## Overview

This project transforms the original XTTS-Server into a Modal-compatible, serverless application that provides:

- **Serverless Architecture**: Auto-scaling GPU containers that spin up on demand
- **Public API Endpoints**: RESTful API accessible from anywhere on the internet
- **Persistent Storage**: Modal Volumes for model weights and speaker data
- **GPU Acceleration**: Automatic GPU provisioning (A10G recommended)
- **High Performance**: Memory snapshots for faster cold boot times

## Features

- ✅ **Health Check Endpoint**: `/healthz` for service monitoring
- ✅ **Speaker Registration**: `/register_speaker` for managing voice references
- ✅ **Text-to-Speech Synthesis**: `/tts` for generating speech from text
- ✅ **Multiple Input Methods**: Support for URLs, base64, and speaker IDs
- ✅ **Concurrent Processing**: Up to 10 concurrent requests per container
- ✅ **API Key Authentication**: Optional security with environment variables
- ✅ **Streaming Responses**: Efficient audio delivery

## Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal setup
   ```
3. **Hugging Face Model**: A fine-tuned XTTS v2 model available on Hugging Face Hub (e.g., `Genarabia-ai/Kuwaiti_XTTS_Latest`).

### Installation

1. **Clone this repository**:
   ```bash
   git clone <your-repo-url>
   cd xtts-modal-server
   ```

2. **Download and upload your model files from Hugging Face**:
   ```bash
   python setup_volumes.py <hugging_face_model_id> [optional_api_key]
   ```
   For example:
   ```bash
   python setup_volumes.py Genarabia-ai/Kuwaiti_XTTS_Latest my_secret_api_key
   ```

3. **Deploy to Modal** (Use the fixed version):
   ```bash
   modal deploy modal_xtts_server_fixed.py
   ```

4. **Get your endpoint URL** from the Modal dashboard or deployment output.

## API Documentation

### Base URL
Your deployed Modal app will have a URL like:
```
https://your-workspace--xtts-server-xttsservice-tts.modal.run
```

### Endpoints

#### Health Check
```http
GET /healthz
```
Returns `200 OK` with "ok" response.

#### Root Information
```http
GET /
```
Returns API information and available endpoints.

#### Register Speaker
```http
POST /register_speaker
Content-Type: application/json
x-api-key: your_api_key (if configured)

{
  "speaker_wav_url": "https://example.com/speaker.wav"
}
```

Or with base64:
```json
{
  "speaker_wav_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
}
```

**Response**:
```json
{
  "speaker_id": "a1b2c3d4e5f6a7b8",
  "path": "/speakers/a1b2c3d4e5f6a7b8.wav"
}
```

#### Text-to-Speech Synthesis
```http
POST /tts
Content-Type: application/json
x-api-key: your_api_key (if configured)

{
  "text": "مرحبا، كيف حالك؟",
  "language": "ar",
  "speaker_id": "a1b2c3d4e5f6a7b8",
  "temperature": 0.75,
  "return_base64": false
}
```

**Parameters**:
- `text` (required): Text to synthesize
- `language` (optional): Language code (default: "ar")
- `speaker_id` (optional): Registered speaker ID
- `speaker_wav_url` (optional): Direct speaker audio URL
- `speaker_wav_base64` (optional): Direct speaker audio as base64
- `temperature` (optional): Synthesis temperature (default: 0.75)
- `return_base64` (optional): Return audio as base64 JSON (default: false)
- `sample_rate` (optional): Output sample rate (default: 24000)

**Response**: 
- Audio stream (WAV file) if `return_base64=false`
- JSON with base64 audio if `return_base64=true`

## Usage Examples

### Python Client
```python
import requests

# Health check
response = requests.get("https://your-modal-url/healthz")
print(response.text)  # "ok"

# Register speaker
speaker_data = {
    "speaker_wav_url": "https://example.com/voice.wav"
}
response = requests.post(
    "https://your-modal-url/register_speaker",
    json=speaker_data,
    headers={"x-api-key": "your_api_key"}
)
speaker_id = response.json()["speaker_id"]

# Generate speech
tts_data = {
    "text": "Hello, this is a test.",
    "language": "en",
    "speaker_id": speaker_id
}
response = requests.post(
    "https://your-modal-url/tts",
    json=tts_data,
    headers={"x-api-key": "your_api_key"}
)

# Save audio file
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL Examples
```bash
# Health check
curl https://your-modal-url/healthz

# Register speaker
curl -X POST "https://your-modal-url/register_speaker" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{"speaker_wav_url": "https://example.com/voice.wav"}'

# Generate speech
curl -X POST "https://your-modal-url/tts" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "text": "مرحبا بك في خدمة التحويل النصي إلى كلام",
    "language": "ar",
    "speaker_id": "your_speaker_id"
  }' \
  --output output.wav
```

## ✅ Verified Testing Examples

### Test Arabic TTS (Verified Working)
```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fixed-xttsservice-tts.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا، هذا اختبار لنموذج الكويتي للتحويل من النص إلى الكلام",
    "language": "ar",
    "speaker_id": "85f343e9362c0bbf",
    "temperature": 0.75
  }' \
  --output arabic_speech.wav
```

### Test English TTS (Verified Working)
```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fixed-xttsservice-tts.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Kuwaiti XTTS model",
    "language": "en",
    "speaker_id": "85f343e9362c0bbf",
    "return_base64": true
  }'
```

### Register Speaker (Verified Working)
```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fixed-xttsservice-register-speaker.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_wav_url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Allah_Wish.wav"
  }'
```

## Development

### Local Testing
```bash
# Serve locally for development
modal serve modal_xtts_server_fixed.py

# Run tests against deployed server
python test_modal_server.py https://your-modal-url your_api_key test_audio.wav
```

### File Structure
```
xtts-modal-server/
├── modal_xtts_server.py           # Original Modal application
├── modal_xtts_server_direct.py    # Direct HF download version
├── modal_xtts_server_fixed.py     # Fixed version (RECOMMENDED)
├── setup_volumes.py               # Script to download and upload model files
├── test_modal_server.py           # Test suite for the API
├── requirements.txt               # Python dependencies
├── deployment_summary.md          # Deployment and testing summary
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Configuration

### Environment Variables
Set these in Modal Secrets or environment:

- `API_KEY`: Optional API key for authentication
- Default values are configured in the code for other settings

### Modal Resources
- **GPU**: A10G (recommended) or other CUDA-compatible GPUs
- **Memory**: Automatic based on model size
- **Storage**: Modal Volumes for persistent data
- **Concurrency**: Up to 10 requests per container

## Performance

### Cold Boot Time
- ~30-60 seconds (with memory snapshots enabled)
- Subsequent requests: ~2-5 seconds

### Scaling
- Automatic scaling based on demand
- Containers kept warm for 5 minutes after last request
- Multiple containers can run in parallel for high load

## Troubleshooting

### Common Issues

1. **"'GPT2InferenceModel' object has no attribute 'generate'" Error**:
   - **SOLUTION**: Use `modal_xtts_server_fixed.py` which pins transformers to `<4.50`
   - This is a known compatibility issue with transformers v4.50+
   - The fixed version resolves this by using compatible library versions

2. **Model not found error**:
   - Ensure model files are downloaded and uploaded to the `xtts-model` volume using `setup_volumes.py`
   - Check file names: `config.json`, `model.pth`, `vocab.json`

3. **GPU memory errors**:
   - Try reducing concurrency or using a larger GPU
   - Check model size compatibility

4. **Authentication errors**:
   - Verify API key is set correctly in Modal Secrets
   - Check `x-api-key` header in requests

5. **Audio processing errors**:
   - Ensure audio files are valid WAV/MP3 format
   - Check audio file accessibility (URLs)

### Logs and Monitoring
- View logs in Modal dashboard
- Use `/healthz` endpoint for monitoring
- Check container metrics in Modal console

## Migration from Original XTTS-Server

### Key Differences
- **Serverless**: No always-on server, functions run on demand
- **Storage**: Modal Volumes instead of local filesystem
- **Scaling**: Automatic instead of manual concurrency control
- **Deployment**: Modal instead of Docker/Vast.ai

### Migration Steps
1. Get your Hugging Face model repository ID
2. Upload using `setup_volumes.py`
3. Deploy the Modal application
4. Update client code to use new endpoint URLs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `test_modal_server.py`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review Modal documentation at [docs.modal.com](https://docs.modal.com)
- Open an issue in this repository

## Acknowledgments

- Original XTTS-Server by [Nourahmed113](https://github.com/Nourahmed113/XTTS-Server)
- Coqui TTS team for the XTTS model
- Modal Labs for the serverless platform
- Genarabia-ai for the Kuwaiti XTTS model

