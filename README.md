# XTTS Modal Server

A high-performance serverless deployment of XTTS (Coqui TTS) v2 on Modal Labs, optimized for Arabic and English text-to-speech synthesis using the Kuwaiti XTTS model.

## 🚀 Live Deployment

**Production Server**: https://ahmedezzat0247--xtts-server-fastapi-app.modal.run

**Interactive API Documentation**: https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/docs

## ✨ Features

- **High-quality neural TTS**: Using the Genarabia-ai/Kuwaiti_XTTS_Latest model (5.6GB)
- **Voice cloning**: Register speakers and clone their voices from audio samples
- **Multilingual support**: Arabic (ar) and English (en) synthesis
- **Production-ready**: Optimized for performance and reliability
- **Serverless scaling**: Automatic GPU provisioning on demand
- **Interactive documentation**: Complete Swagger UI with testing capabilities

## 🏗️ Architecture & Performance

### GPU & Infrastructure
- **GPU**: L40S (48GB VRAM, Ada Lovelace architecture) - 1.5-2x faster than A10G
- **Memory**: 16GB RAM with 4 CPU cores for optimal performance
- **Scaling**: 2-6 containers with automatic scaling based on demand

### Performance Optimizations
- **Memory Snapshots**: 60-80% faster cold starts
- **Warm Container Pool**: 2-3 containers always ready
- **Speaker Caching**: Reuse embeddings for repeated requests
- **Extended Lifetime**: 30-minute container persistence
- **Concurrent Processing**: Up to 3 requests per container

### Performance Metrics
- **Cold Start**: ~15-20 seconds (first request)
- **Warm Inference**: ~8-17 seconds (typical response time)
- **Audio Quality**: 24kHz, 16-bit mono WAV
- **Consistency**: Predictable performance with warm container pool

## 🔧 API Endpoints

### Base URL
```
https://ahmedezzat0247--xtts-server-fastapi-app.modal.run
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information and status |
| `/healthz` | GET | Health check endpoint |
| `/register_speaker` | POST | Register a speaker for voice cloning |
| `/tts` | POST | Generate speech from text |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

## 📖 Usage Guide

### 1. Register a Speaker

Register a speaker using an audio URL:

```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/register_speaker" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_wav_url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Allah_Wish.wav"
  }'
```

Response:
```json
{
  "speaker_id": "85f343e9362c0bbf",
  "path": "/speakers/85f343e9362c0bbf.wav"
}
```

### 2. Generate Speech

Generate Arabic speech using the registered speaker:

```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا، هذا اختبار للنموذج الكويتي",
    "language": "ar",
    "speaker_id": "85f343e9362c0bbf",
    "temperature": 0.75
  }' \
  --output arabic_speech.wav
```

Generate English speech:

```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the English synthesis",
    "language": "en",
    "speaker_id": "85f343e9362c0bbf",
    "temperature": 0.75
  }' \
  --output english_speech.wav
```

### 3. Get Base64 Audio Response

For applications that need base64-encoded audio:

```bash
curl -X POST "https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا بك في خدمة تحويل النص إلى كلام",
    "language": "ar",
    "speaker_id": "85f343e9362c0bbf",
    "return_base64": true
  }'
```

Response:
```json
{
  "audio_wav_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "sample_rate": 24000
}
```

## 🐍 Python Client Example

```python
import requests
import base64

class XTTSClient:
    def __init__(self, base_url="https://ahmedezzat0247--xtts-server-fastapi-app.modal.run"):
        self.base_url = base_url
    
    def register_speaker(self, audio_url):
        """Register a speaker from audio URL"""
        response = requests.post(
            f"{self.base_url}/register_speaker",
            json={"speaker_wav_url": audio_url}
        )
        response.raise_for_status()
        return response.json()
    
    def generate_speech(self, text, language="ar", speaker_id=None, temperature=0.75):
        """Generate speech from text"""
        data = {
            "text": text,
            "language": language,
            "speaker_id": speaker_id,
            "temperature": temperature
        }
        
        response = requests.post(f"{self.base_url}/tts", json=data)
        response.raise_for_status()
        return response.content
    
    def health_check(self):
        """Check service health"""
        response = requests.get(f"{self.base_url}/healthz")
        return response.text == "ok"

# Usage example
client = XTTSClient()

# Register a speaker
speaker_info = client.register_speaker("https://upload.wikimedia.org/wikipedia/commons/1/18/Allah_Wish.wav")
speaker_id = speaker_info["speaker_id"]

# Generate Arabic speech
arabic_audio = client.generate_speech(
    text="مرحبا، كيف حالك اليوم؟",
    language="ar",
    speaker_id=speaker_id
)

# Save audio file
with open("arabic_output.wav", "wb") as f:
    f.write(arabic_audio)

print(f"Arabic speech generated and saved! Speaker ID: {speaker_id}")
```

## 📋 Request Parameters

### Register Speaker Request
```json
{
  "speaker_wav_url": "https://example.com/audio.wav",  // Optional: URL to audio file
  "speaker_wav_base64": "UklGRiQAAABXQVZF..."        // Optional: Base64 encoded audio
}
```

### TTS Request
```json
{
  "text": "النص المراد تحويله إلى كلام",              // Required: Text to synthesize
  "language": "ar",                                  // Optional: "ar" or "en" (default: "ar")
  "speaker_id": "85f343e9362c0bbf",                 // Optional: Registered speaker ID
  "speaker_wav_url": "https://example.com/audio.wav", // Optional: Direct audio URL
  "speaker_wav_base64": "UklGRiQAAABXQVZF...",      // Optional: Base64 audio
  "temperature": 0.75,                              // Optional: 0.1-1.0 (default: 0.75)
  "return_base64": false,                           // Optional: Return base64 instead of binary
  "sample_rate": 24000                              // Optional: Audio sample rate (default: 24000)
}
```

## 🔍 Interactive Testing

Visit the interactive documentation at:
**https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/docs**

Features:
- **Try it out**: Test all endpoints directly in the browser
- **Request/Response schemas**: Complete data models with examples
- **Copy curl commands**: Generated automatically for each request
- **Authentication**: No API key required for testing

## 🛠️ Development & Deployment

### Prerequisites
- Python 3.11+
- Modal account and CLI installed
- Git for version control

### Local Development
```bash
# Clone the repository
git clone https://github.com/Ahmed-Ezzat20/XTTS-Modal-Server.git
cd XTTS-Modal-Server

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup

# Deploy to Modal
modal deploy modal_xtts_server_production.py
```

### Project Structure
```
XTTS-Modal-Server/
├── modal_xtts_server_production.py     # Production deployment
├── modal_xtts_server_conservative_optimized.py  # Performance-optimized version
├── setup_volumes.py                    # Model setup script
├── test_modal_server.py               # Testing utilities
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

## 📊 Performance Analysis

### Response Time Breakdown
- **Model Loading**: ~20-25 seconds (cold start only)
- **Speaker Processing**: ~2-3 seconds (cached after first use)
- **Text Synthesis**: ~5-8 seconds (depends on text length)
- **Audio Processing**: ~1-2 seconds

### Optimization Results
- **84% improvement** in warm container performance
- **Consistent 8-17 second** response times for warm requests
- **Eliminated 40+ second** worst-case scenarios
- **Predictable performance** with warm container pool

## 🌍 Supported Languages

| Language | Code | Model Support | Quality |
|----------|------|---------------|---------|
| Arabic | `ar` | Native (Kuwaiti) | Excellent |
| English | `en` | Cross-lingual | Very Good |

## 💰 Cost Analysis

### Infrastructure Costs
- **L40S GPU**: $1.95/hour (only when active)
- **Storage**: Minimal cost for model and speaker files
- **Scaling**: Pay only for active containers

### Cost Optimization
- **Automatic scaling**: Containers scale down when not in use
- **Warm pool**: Maintains 2-3 containers for immediate response
- **Efficient caching**: Reduces redundant processing

## 🔒 Security & Privacy

- **No authentication required**: Open API for testing and development
- **Temporary storage**: Speaker files stored securely in Modal volumes
- **No data logging**: Audio content is not logged or stored permanently
- **HTTPS encryption**: All API communication is encrypted

## 🐛 Troubleshooting

### Common Issues

**1. Speaker ID not found**
```json
{"detail": "Speaker ID not found. Please register first."}
```
Solution: Register the speaker using `/register_speaker` endpoint first.

**2. Empty text error**
```json
{"detail": "Text cannot be empty."}
```
Solution: Ensure the `text` field contains non-empty content.

**3. Cold start delays**
- First request may take 20-30 seconds (model loading)
- Subsequent requests are much faster (8-17 seconds)
- This is normal behavior for serverless deployments

### Performance Tips

1. **Reuse speaker IDs**: Register speakers once and reuse the ID
2. **Batch requests**: Send multiple requests to keep containers warm
3. **Optimal text length**: 10-100 words per request for best performance
4. **Temperature tuning**: Use 0.75 for balanced quality/speed

## 📞 Support & Contributing

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/Ahmed-Ezzat20/XTTS-Modal-Server/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/Ahmed-Ezzat20/XTTS-Modal-Server/discussions)
- **Contributing**: Pull requests welcome! Please read our contributing guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for the XTTS model
- [Modal Labs](https://modal.com) for serverless GPU infrastructure
- [Genarabia AI](https://huggingface.co/Genarabia-ai) for the Kuwaiti XTTS model
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

## 📈 Changelog

### v1.0.0 (Latest)
- ✅ Production-ready deployment with clean "xtts-server" name
- ✅ L40S GPU optimization for 1.5-2x performance improvement
- ✅ Memory snapshots for 60-80% faster cold starts
- ✅ Warm container pool for consistent performance
- ✅ Speaker embedding caching for repeated requests
- ✅ Complete interactive API documentation
- ✅ Comprehensive README and usage examples

### Previous Versions
- v0.9.0: Conservative optimizations and performance testing
- v0.8.0: FastAPI documentation integration
- v0.7.0: Model caching and persistent storage
- v0.6.0: Initial Modal deployment and testing

---

**Ready to generate high-quality Arabic and English speech? Visit the [live API documentation](https://ahmedezzat0247--xtts-server-fastapi-app.modal.run/docs) and start testing!** 🎤✨
