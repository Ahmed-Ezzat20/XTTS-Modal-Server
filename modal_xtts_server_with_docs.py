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
app = modal.App("xtts-server-with-docs", image=image)

# Define volume for persistent speaker storage
speaker_volume = modal.Volume.from_name("xtts-speakers", create_if_missing=True)

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


# Main TTS service class
@app.cls(
    gpu="a10g",
    volumes={
        "/speakers": speaker_volume
    },
    scaledown_window=60 * 5,  # Keep containers alive for 5 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots for faster cold boots
    timeout=1800,  # 30 minutes timeout for model loading
)
@modal.concurrent(max_inputs=10)  # Allow up to 10 concurrent requests per container
class XTTSService:
    
    @modal.enter()
    def load_model(self):
        """Load the XTTS model when the container starts"""
        print("Loading XTTS model from Hugging Face...")
        
        # Model repository ID
        model_repo_id = "Genarabia-ai/Kuwaiti_XTTS_Latest"
        
        # Create a temporary directory for model files
        model_dir = "/tmp/xtts_model"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Download model files from Hugging Face
            print("Downloading config.json...")
            config_path = hf_hub_download(repo_id=model_repo_id, filename="config.json", cache_dir=model_dir)
            
            print("Downloading model.pth...")
            ckpt_path = hf_hub_download(repo_id=model_repo_id, filename="model.pth", cache_dir=model_dir)
            
            print("Downloading vocab.json...")
            try:
                vocab_path = hf_hub_download(repo_id=model_repo_id, filename="vocab.json", cache_dir=model_dir)
            except:
                print("vocab.json not found, using default")
                vocab_path = None
            
            print("Model files downloaded successfully!")
            
            # Load model configuration
            config = XttsConfig()
            config.load_json(config_path)
            
            # Initialize and load the model
            model = Xtts.init_from_config(config)
            
            # Get the actual directory containing the model files
            model_files_dir = os.path.dirname(ckpt_path)
            
            model.load_checkpoint(
                config,
                checkpoint_dir=model_files_dir,
                vocab_path=vocab_path,
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
            
            print(f"XTTS model loaded successfully on {device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def _require_api_key(self, x_api_key: Optional[str] = None):
        """API key authentication disabled"""
        # No authentication required
        pass
    
    def _ensure_speakers_directory(self):
        """Ensure the speakers directory exists with proper permissions"""
        speakers_dir = "/speakers"
        if not os.path.exists(speakers_dir):
            os.makedirs(speakers_dir, mode=0o755, exist_ok=True)
        return speakers_dir
    
    async def _get_audio_bytes(self, req: RegisterSpeakerRequest) -> bytes:
        """Download or decode audio from URL or base64"""
        if req.speaker_wav_base64:
            try:
                return base64.b64decode(req.speaker_wav_base64)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 audio data.")
        elif req.speaker_wav_url:
            import httpx
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(str(req.speaker_wav_url))
                    response.raise_for_status()
                    return response.content
                except httpx.HTTPError as e:
                    raise HTTPException(
                        status_code=400, detail=f"Failed to download audio: {e}"
                    )
        else:
            raise HTTPException(
                status_code=400, detail="Either speaker_wav_url or speaker_wav_base64 must be provided"
            )
    
    async def _resolve_speaker_path(self, req: TTSRequest) -> Optional[str]:
        """Resolve speaker audio path from various input methods"""
        if req.speaker_id:
            path = f"/speakers/{req.speaker_id}.wav"
            if not os.path.exists(path):
                raise HTTPException(
                    status_code=404, detail="Speaker ID not found. Please register first."
                )
            return path
        
        if req.speaker_wav_base64 or req.speaker_wav_url:
            # Create temporary speaker from provided audio
            audio_req = RegisterSpeakerRequest(
                speaker_wav_url=req.speaker_wav_url,
                speaker_wav_base64=req.speaker_wav_base64,
            )
            audio_bytes = await self._get_audio_bytes(audio_req)
            speaker_id = hashlib.sha1(audio_bytes).hexdigest()[:16]
            save_path = f"/speakers/{speaker_id}.wav"
            
            try:
                with io.BytesIO(audio_bytes) as buf:
                    data, sr = sf.read(buf, dtype="float32", always_2d=False)
                    
                    # Ensure directory exists before writing
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sf.write(save_path, data, sr, format="WAV")
                
                # Commit the volume to persist the speaker file
                speaker_volume.commit()
                return save_path
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process speaker audio: {e}"
                )
        
        return None


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
    
    ## Usage
    1. **Register a speaker** using `/register_speaker` with an audio URL or base64 data
    2. **Generate speech** using `/tts` with text, language, and speaker ID
    3. **Monitor health** using `/healthz` for service status
    
    ## Model Information
    - **Model**: Genarabia-ai/Kuwaiti_XTTS_Latest (5.6GB)
    - **Languages**: Arabic (ar), English (en)
    - **Audio Quality**: 24kHz, 16-bit mono WAV
    - **Processing Speed**: ~1.6x real-time generation
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
        service="XTTS v2 Inference API on Modal (With Documentation)",
        model="Genarabia-ai/Kuwaiti_XTTS_Latest",
        endpoints=["/healthz", "/tts", "/register_speaker", "/docs", "/redoc"],
        note="Use /tts (POST) for synthesis. Use /register_speaker (POST) to register speakers. Visit /docs for interactive API documentation.",
        version="fixed-transformers-compatibility-with-docs"
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
    
    # Ensure speakers directory exists
    xtts_service._ensure_speakers_directory()
    
    # Get audio data
    audio_bytes = await xtts_service._get_audio_bytes(req)
    speaker_id = hashlib.sha1(audio_bytes).hexdigest()[:16]
    save_path = f"/speakers/{speaker_id}.wav"
    
    try:
        # Process and save audio
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
            
            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sf.write(save_path, data, sr, format="WAV")
        
        # Commit the volume to persist the speaker file
        speaker_volume.commit()
        
        return RegisterSpeakerResponse(speaker_id=speaker_id, path=save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")


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
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Resolve speaker audio path
    speaker_path = await xtts_service._resolve_speaker_path(req)
    
    try:
        # Get conditioning latents from speaker audio
        gpt_cond_latent, speaker_embedding = (
            xtts_service.model.get_conditioning_latents(audio_path=[speaker_path])
            if speaker_path
            else (None, None)
        )
        
        # Generate speech
        with torch.no_grad():
            out = xtts_service.model.inference(
                text=req.text,
                language=req.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=req.temperature,
            )
            
            # Extract audio data (check for 'wav' or 'audio' key)
            wav_np = out.get("wav")
            if wav_np is None:
                wav_np = out.get("audio")
            
            if wav_np is None:
                raise HTTPException(status_code=500, detail="No audio output generated")
        
        # Convert to numpy if tensor
        if isinstance(wav_np, torch.Tensor):
            wav_np = wav_np.cpu().numpy()
        
        # Create audio buffer
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, wav_np.astype("float32"), req.sample_rate, format="WAV")
        wav_bytes.seek(0)
        
        # Return response based on request format
        if req.return_base64:
            b64_audio = base64.b64encode(wav_bytes.read()).decode("ascii")
            return JSONResponse(
                {"audio_wav_base64": b64_audio, "sample_rate": req.sample_rate}
            )
        
        return StreamingResponse(wav_bytes, media_type="audio/wav")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# Deploy the FastAPI app
@app.function(image=image, timeout=1800)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    # This allows running the script locally for testing
    print("XTTS Modal server with documentation ready for deployment!")
    print("Deploy with: modal deploy modal_xtts_server_with_docs.py")
    print("After deployment, visit /docs for interactive API documentation")
