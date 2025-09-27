import asyncio
import base64
import hashlib
import io
import os
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

# Create the Modal app
app = modal.App("xtts-server-optimized", image=image)

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
        schema_extra = {
            "example": {
                "speaker_wav_url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Allah_Wish.wav"
            }
        }


class RegisterSpeakerResponse(BaseModel):
    """Response model for speaker registration."""
    speaker_id: str
    path: str
    
    class Config:
        schema_extra = {
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
        schema_extra = {
            "example": {
                "text": "مرحبا، هذا اختبار للنموذج الكويتي للتحويل من النص إلى الكلام",
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


# Model setup function (runs once to download and cache the model)
@app.function(
    image=image,
    volumes={"/model": model_volume},
    timeout=1800
)
def setup_model():
    """Download and cache the XTTS model to persistent storage"""
    
    model_dir = "/model"
    model_repo_id = "Genarabia-ai/Kuwaiti_XTTS_Latest"
    
    # Check if model is already downloaded
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.pth")
    
    if os.path.exists(config_path) and os.path.exists(model_path):
        print("Model already exists in persistent storage!")
        return "Model already cached"
    
    print("Downloading XTTS model to persistent storage...")
    
    try:
        # Download model files to persistent volume
        print("Downloading config.json...")
        hf_hub_download(
            repo_id=model_repo_id, 
            filename="config.json", 
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print("Downloading model.pth...")
        hf_hub_download(
            repo_id=model_repo_id, 
            filename="model.pth", 
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print("Downloading vocab.json...")
        try:
            hf_hub_download(
                repo_id=model_repo_id, 
                filename="vocab.json", 
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
        except:
            print("vocab.json not found, using default")
        
        # Commit the volume to persist the model files
        model_volume.commit()
        
        print("Model downloaded and cached successfully!")
        return "Model downloaded and cached"
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise e


# Main TTS service class with persistent model loading
@app.cls(
    gpu="a10g",
    volumes={
        "/speakers": speaker_volume,
        "/model": model_volume
    },
    scaledown_window=60 * 10,  # Keep containers alive for 10 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots for faster cold boots
    timeout=1800,  # 30 minutes timeout
)
@modal.concurrent(max_inputs=10)  # Allow up to 10 concurrent requests per container
class XTTSService:
    
    @modal.enter()
    def load_model(self):
        """Load the XTTS model from persistent storage"""
        print("Loading XTTS model from persistent storage...")
        
        model_dir = "/model"
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.pth")
        vocab_path = os.path.join(model_dir, "vocab.json")
        
        # Check if model files exist
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise Exception("Model files not found in persistent storage. Run setup_model() first.")
        
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
            
            # Move model to GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            # Store in instance variables
            self.model = model
            self.config = config
            self.device = device
            
            print(f"XTTS model loaded successfully from persistent storage on {device}")
            
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
        """Generate TTS audio using the loaded model"""
        
        # Check if speaker file exists
        if not os.path.exists(speaker_path):
            raise Exception("Speaker ID not found. Please register first.")
        
        try:
            # Get conditioning latents from speaker audio
            gpt_cond_latent, speaker_embedding = (
                self.model.get_conditioning_latents(audio_path=[speaker_path])
            )
            
            # Generate speech
            with torch.no_grad():
                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                )
                
                # Extract audio data (check for 'wav' or 'audio' key)
                wav_np = out.get("wav")
                if wav_np is None:
                    wav_np = out.get("audio")
                
                if wav_np is None:
                    raise Exception("No audio output generated")
            
            # Convert to numpy if tensor
            if isinstance(wav_np, torch.Tensor):
                wav_np = wav_np.cpu().numpy()
            
            # Create audio buffer
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, wav_np.astype("float32"), sample_rate, format="WAV")
            wav_bytes.seek(0)
            
            return wav_bytes.read()
            
        except Exception as e:
            raise Exception(f"Inference failed: {e}")


# Speaker registration function
@app.function(
    image=image,
    volumes={"/speakers": speaker_volume},
    timeout=120
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
        
        async with httpx.AsyncClient() as client:
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
            sf.write(save_path, data, sr, format="WAV")
        
        # Commit the volume to persist the speaker file
        speaker_volume.commit()
        
        return {"speaker_id": speaker_id, "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")


# Create FastAPI app with documentation
web_app = FastAPI(
    title="XTTS v2 Inference API",
    description="""
    A serverless deployment of XTTS (Coqui TTS) v2 on Modal Labs platform with public API endpoints for text-to-speech synthesis.
    
    ## Features
    - **High-quality neural TTS**: Using the Kuwaiti XTTS model
    - **Voice cloning**: Register speakers and clone their voices
    - **Multilingual**: Supports Arabic and English synthesis
    - **Serverless scaling**: Automatic GPU provisioning on demand
    - **Fast generation**: ~1.6x real-time synthesis speed
    - **Persistent model caching**: Model downloaded once and cached
    
    ## Usage
    1. **Register a speaker** using `/register_speaker` with an audio URL or base64 data
    2. **Generate speech** using `/tts` with text, language, and speaker ID
    3. **Monitor health** using `/healthz` for service status
    
    ## Model Information
    - **Model**: Genarabia-ai/Kuwaiti_XTTS_Latest (5.6GB)
    - **Languages**: Arabic (ar), English (en)
    - **Audio Quality**: 24kHz, 16-bit mono WAV
    - **Processing Speed**: ~1.6x real-time generation
    - **Storage**: Persistent model and speaker caching
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
        service="XTTS v2 Inference API on Modal (Optimized with Persistent Storage)",
        model="Genarabia-ai/Kuwaiti_XTTS_Latest",
        endpoints=["/healthz", "/tts", "/register_speaker", "/docs", "/redoc", "/setup"],
        note="Use /tts (POST) for synthesis. Use /register_speaker (POST) to register speakers. Visit /docs for interactive API documentation.",
        version="optimized-with-persistent-model-storage"
    )


@web_app.get("/healthz", tags=["Health"])
async def healthz():
    """Health check endpoint. Returns 'ok' if the service is running."""
    return PlainTextResponse("ok")


@web_app.post("/setup", tags=["Setup"])
async def setup():
    """Download and cache the model to persistent storage (run once)."""
    try:
        result = await setup_model.remote.aio()
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {e}")


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
    - Processing time: ~4-10 seconds depending on text length
    - Generation speed: ~1.6x real-time (faster than playback)
    - Audio quality: 24kHz, 16-bit mono WAV
    - Model loading: Cached in persistent storage (no re-download)
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Resolve speaker audio path
    speaker_path = None
    
    if req.speaker_id:
        speaker_path = f"/speakers/{req.speaker_id}.wav"
        # Check if speaker exists by calling the service
        try:
            # Use the service to check if speaker exists and generate TTS
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
    # This allows running the script locally for testing
    print("XTTS Modal server with optimized persistent storage ready for deployment!")
    print("Deploy with: modal deploy modal_xtts_server_optimized.py")
    print("After deployment:")
    print("1. Run /setup endpoint once to download and cache the model")
    print("2. Visit /docs for interactive API documentation")
    print("3. Use /register_speaker and /tts for normal operations")
