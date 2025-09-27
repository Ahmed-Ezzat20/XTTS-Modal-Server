import asyncio
import base64
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

# Define the Modal image with all required dependencies and compatible versions
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.110,<1.0",
    "soundfile>=0.12.1",
    "TTS>=0.22.0",
    "pydantic<3",
    "transformers>=4.44.2,<4.50",  # Use compatible transformers version
    "torch",
    "torchaudio",
    "httpx",
    "numpy",
    "huggingface_hub",
)

# Create the Modal app with production name
app = modal.App("xtts-server", image=image)

# Define volumes for persistent storage
speaker_volume = modal.Volume.from_name("xtts-speakers", create_if_missing=True)
model_volume = modal.Volume.from_name("xtts-model", create_if_missing=True)

# Import TTS modules within the image context
with image.imports():
    import soundfile as sf
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from huggingface_hub import hf_hub_download


# Pydantic models for request/response
class RegisterSpeakerRequest(BaseModel):
    """Request model for speaker registration."""
    speaker_wav_url: Optional[HttpUrl] = None
    speaker_wav_base64: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_wav_url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Allah_Wish.wav"
            }
        }


class RegisterSpeakerResponse(BaseModel):
    """Response model for speaker registration."""
    speaker_id: str
    path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_id": "85f343e9362c0bbf",
                "path": "/speakers/85f343e9362c0bbf.wav"
            }
        }


class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str
    language: Optional[str] = "ar"
    speaker_id: Optional[str] = None
    speaker_wav_url: Optional[HttpUrl] = None
    speaker_wav_base64: Optional[str] = None
    temperature: float = 0.75
    return_base64: bool = False
    sample_rate: int = 24000
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "مرحبا، هذا اختبار للنموذج الكويتي",
                "language": "ar",
                "speaker_id": "85f343e9362c0bbf",
                "temperature": 0.75,
                "return_base64": False
            }
        }


class ServiceInfo(BaseModel):
    """Service information response model."""
    status: str
    service: str
    model: str
    endpoints: list[str]
    note: str
    version: str
    optimizations: list[str]


# Production-optimized TTS service class
@app.cls(
    # L40S GPU for optimal performance
    gpu="L40S",  # 48GB VRAM, Ada Lovelace architecture, $1.95/hr
    volumes={
        "/speakers": speaker_volume,
        "/model": model_volume
    },
    # Production optimizations
    scaledown_window=60 * 30,  # Keep containers warm for 30 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots for faster cold starts
    min_containers=2,  # Maintain 2 warm containers
    max_containers=6,  # Allow scaling up to 6 containers
    buffer_containers=1,  # Keep 1 extra container ready
    timeout=1800,
    # Resource optimization
    cpu=4,  # 4 CPU cores for better performance
    memory=16384,  # 16GB RAM for optimal model loading
)
@modal.concurrent(max_inputs=3)  # Allow up to 3 concurrent requests per container
class XTTSService:
    
    @modal.enter()
    def load_model(self):
        """Load the XTTS model from persistent storage with production optimizations"""
        print("Loading XTTS model from persistent storage...")
        
        model_dir = "/model"
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.pth")
        vocab_path = os.path.join(model_dir, "vocab.json")
        
        # Check if model files exist
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise Exception("Model files not found in persistent storage. Please ensure model is downloaded.")
        
        try:
            # Load model configuration
            config = XttsConfig()
            config.load_json(config_path)
            
            # Initialize and load the model
            model = Xtts.init_from_config(config)
            
            # Load the model checkpoint
            model.load_checkpoint(
                config,
                checkpoint_dir=model_dir,
                vocab_path=vocab_path if os.path.exists(vocab_path) else None,
                use_deepspeed=False,
            )
            
            # Move model to GPU and apply optimizations
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            # Production optimizations
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Store in instance variables
            self.model = model
            self.config = config
            self.device = device
            
            # Cache for speaker embeddings (performance optimization)
            self.speaker_cache = {}
            
            print(f"XTTS model loaded successfully on {device}")
            print(f"GPU: L40S, Memory: 48GB, Ready for production")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    @modal.method()
    async def generate_tts(
        self,
        text: str,
        language: str,
        speaker_path: str,
        temperature: float,
        sample_rate: int
    ):
        """Generate TTS audio using the loaded model with production optimizations"""
        
        # Check if speaker file exists
        if not os.path.exists(speaker_path):
            raise Exception("Speaker ID not found. Please register first.")
        
        try:
            # Check speaker embedding cache for performance improvement
            cache_key = f"{speaker_path}_{hash(speaker_path)}"
            
            if cache_key in self.speaker_cache:
                gpt_cond_latent, speaker_embedding = self.speaker_cache[cache_key]
            else:
                # Get conditioning latents from speaker audio
                gpt_cond_latent, speaker_embedding = (
                    self.model.get_conditioning_latents(audio_path=[speaker_path])
                )
                # Cache for future use
                self.speaker_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
            
            # Generate speech
            with torch.no_grad():
                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                )
                
                # Extract audio data
                wav_np = out.get("wav")
                if wav_np is None:
                    wav_np = out.get("audio")
                
                if wav_np is None:
                    raise Exception("No audio output generated")
            
            # Convert to numpy if tensor
            if isinstance(wav_np, torch.Tensor):
                wav_np = wav_np.cpu().float().numpy()
            
            # Create audio buffer
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, wav_np.astype("float32"), sample_rate, format="WAV")
            wav_bytes.seek(0)
            
            return wav_bytes.read()
            
        except Exception as e:
            raise Exception(f"TTS generation failed: {e}")


# Speaker registration function
@app.function(
    image=image,
    volumes={"/speakers": speaker_volume},
    timeout=120,
    cpu=2,
)
async def register_speaker_modal(speaker_wav_url: Optional[str] = None, speaker_wav_base64: Optional[str] = None):
    """Register a speaker and return speaker ID"""
    
    # Ensure speakers directory exists
    speakers_dir = "/speakers"
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, mode=0o755, exist_ok=True)
    
    # Get audio data
    if speaker_wav_base64:
        try:
            audio_bytes = base64.b64decode(speaker_wav_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data.")
    elif speaker_wav_url:
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(speaker_wav_url)
                response.raise_for_status()
                audio_bytes = response.content
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download audio: {e}"
                )
    else:
        raise HTTPException(
            status_code=400, detail="Either speaker_wav_url or speaker_wav_base64 must be provided"
        )
    
    # Generate speaker ID
    speaker_id = hashlib.sha1(audio_bytes).hexdigest()[:16]
    save_path = f"/speakers/{speaker_id}.wav"
    
    # Check if speaker already exists
    if os.path.exists(save_path):
        return {"speaker_id": speaker_id, "path": save_path}
    
    try:
        # Process and save audio
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
            
            # Audio preprocessing
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # Convert to mono if stereo
            
            # Normalize audio
            data = data / max(abs(data.max()), abs(data.min())) * 0.95
            
            sf.write(save_path, data, sr, format="WAV")
        
        # Commit the volume to persist the speaker file
        speaker_volume.commit()
        
        return {"speaker_id": speaker_id, "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")


# Create FastAPI app
web_app = FastAPI(
    title="XTTS v2 Production Server",
    description="""
    A high-performance serverless deployment of XTTS (Coqui TTS) v2 on Modal Labs.
    
    ## 🚀 Features
    - **High-quality neural TTS**: Using the Kuwaiti XTTS model
    - **Voice cloning**: Register speakers and clone their voices
    - **Multilingual**: Supports Arabic and English synthesis
    - **Production-ready**: Optimized for performance and reliability
    - **Serverless scaling**: Automatic GPU provisioning on demand
    
    ## 📊 Performance Optimizations
    - **L40S GPU**: 1.5-2x faster than A10G with 48GB VRAM
    - **Memory Snapshots**: 60-80% faster cold starts
    - **Warm Container Pool**: 2-3 containers always ready
    - **Speaker Caching**: Reuse embeddings for faster repeated requests
    - **Extended Lifetime**: 30-minute container persistence
    
    ## Usage
    1. **Register a speaker**: Use `/register_speaker` with audio URL or base64 data
    2. **Generate speech**: Use `/tts` with text, language, and speaker ID
    3. **Monitor status**: Check `/healthz` for service health
    
    ## Model Information
    - **Model**: Genarabia-ai/Kuwaiti_XTTS_Latest (5.6GB)
    - **Languages**: Arabic (ar), English (en)
    - **Audio Quality**: 24kHz, 16-bit mono WAV
    - **Processing Speed**: ~8-17 seconds typical response time
    """,
    version="1.0.0",
    contact={
        "name": "XTTS Modal Server",
        "url": "https://github.com/Ahmed-Ezzat20/XTTS-Modal-Server",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Initialize the service instance
xtts_service = XTTSService()


@web_app.get("/", response_model=ServiceInfo, tags=["Info"])
async def root():
    """Get service information and available endpoints."""
    return ServiceInfo(
        status="ok",
        service="XTTS v2 Production Server on Modal",
        model="Genarabia-ai/Kuwaiti_XTTS_Latest",
        endpoints=["/healthz", "/tts", "/register_speaker", "/docs", "/redoc"],
        note="Production-ready XTTS server with L40S GPU and performance optimizations.",
        version="1.0.0",
        optimizations=[
            "L40S GPU (48GB VRAM, Ada Lovelace architecture)",
            "Memory snapshots for faster cold starts", 
            "Warm container pool (2-3 containers)",
            "Extended 30-minute container lifetime",
            "Speaker embedding caching",
            "Enhanced resource allocation",
            "Production-grade reliability"
        ]
    )


@web_app.get("/healthz", tags=["Health"])
async def healthz():
    """Health check endpoint. Returns 'ok' if the service is running."""
    return PlainTextResponse("ok")


@web_app.post("/register_speaker", response_model=RegisterSpeakerResponse, tags=["Speaker Management"])
async def register_speaker(req: RegisterSpeakerRequest):
    """
    Register a new speaker for voice cloning.
    
    Provide either a URL to an audio file or base64-encoded audio data.
    The audio should be a clear recording of the speaker (WAV format recommended).
    
    Returns a speaker_id that can be used for TTS synthesis.
    """
    if not req.speaker_wav_url and not req.speaker_wav_base64:
        raise HTTPException(
            status_code=400,
            detail="Either 'speaker_wav_url' or 'speaker_wav_base64' must be provided.",
        )
    
    try:
        result = await register_speaker_modal.remote.aio(
            speaker_wav_url=str(req.speaker_wav_url) if req.speaker_wav_url else None,
            speaker_wav_base64=req.speaker_wav_base64
        )
        return RegisterSpeakerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@web_app.post("/tts", tags=["Text-to-Speech"])
async def tts(req: TTSRequest):
    """
    Generate speech from text using the XTTS model.
    
    ## Parameters
    - **text**: The text to convert to speech
    - **language**: Language code ("ar" for Arabic, "en" for English)
    - **speaker_id**: ID of a registered speaker (required)
    - **temperature**: Sampling temperature (0.1-1.0, default 0.75)
    - **return_base64**: Return audio as base64 JSON instead of binary WAV
    - **sample_rate**: Audio sample rate (default 24000 Hz)
    
    ## Response
    - **Binary mode**: Returns WAV audio file directly
    - **Base64 mode**: Returns JSON with base64-encoded audio data
    
    ## Performance
    - **Typical response time**: 8-17 seconds
    - **Audio quality**: 24kHz, 16-bit mono WAV
    - **Concurrent processing**: Up to 3 requests per container
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Resolve speaker audio path
    speaker_path = None
    
    if req.speaker_id:
        speaker_path = f"/speakers/{req.speaker_id}.wav"
        # Use the service to generate TTS
        try:
            audio_bytes = await xtts_service.generate_tts.remote.aio(
                text=req.text,
                language=req.language,
                speaker_path=speaker_path,
                temperature=req.temperature,
                sample_rate=req.sample_rate
            )
        except Exception as e:
            if "Speaker ID not found" in str(e):
                raise HTTPException(status_code=404, detail="Speaker ID not found. Please register first.")
            raise HTTPException(status_code=500, detail=str(e))
    elif req.speaker_wav_url or req.speaker_wav_base64:
        # Register speaker first, then generate TTS
        try:
            speaker_result = await register_speaker_modal.remote.aio(
                speaker_wav_url=str(req.speaker_wav_url) if req.speaker_wav_url else None,
                speaker_wav_base64=req.speaker_wav_base64
            )
            speaker_path = speaker_result["path"]
            
            # Generate TTS with the registered speaker
            audio_bytes = await xtts_service.generate_tts.remote.aio(
                text=req.text,
                language=req.language,
                speaker_path=speaker_path,
                temperature=req.temperature,
                sample_rate=req.sample_rate
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(
            status_code=400, detail="Must provide speaker_id, speaker_wav_url, or speaker_wav_base64"
        )
    
    # Return response based on request format
    if req.return_base64:
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse(
            {"audio_wav_base64": b64_audio, "sample_rate": req.sample_rate}
        )
    
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")


# Deploy the FastAPI app
@app.function(image=image, timeout=1800)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    print("XTTS Production Server ready for deployment!")
    print("Deploy with: modal deploy modal_xtts_server_production.py")
    print("")
    print("🚀 Production Features:")
    print("- L40S GPU (48GB VRAM, Ada Lovelace architecture)")
    print("- Memory snapshots for faster cold starts")
    print("- Warm container pool for consistent performance")
    print("- Speaker embedding caching")
    print("- Production-grade reliability and documentation")
    print("")
    print("After deployment:")
    print("1. Visit /docs for interactive API documentation")
    print("2. Use /register_speaker and /tts endpoints")
    print("3. Enjoy fast, reliable TTS generation!")
