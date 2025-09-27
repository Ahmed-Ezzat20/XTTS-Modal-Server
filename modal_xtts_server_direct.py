import asyncio
import base64
import hashlib
import io
import os
from typing import Optional

import modal
from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

# Define the Modal image with all required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.110,<1.0",
    "soundfile>=0.12.1",
    "TTS>=0.22.0",
    "pydantic<3",
    "transformers>=4.44.2",
    "torch",
    "torchaudio",
    "httpx",
    "numpy",
    "huggingface_hub",
)

# Create the Modal app
app = modal.App("xtts-server", image=image)

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
    speaker_wav_url: Optional[HttpUrl] = None
    speaker_wav_base64: Optional[str] = None


class RegisterSpeakerResponse(BaseModel):
    speaker_id: str
    path: str


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "ar"
    speaker_id: Optional[str] = None
    speaker_wav_url: Optional[HttpUrl] = None
    speaker_wav_base64: Optional[str] = None
    temperature: float = 0.75
    return_base64: bool = False
    sample_rate: int = 24000


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
                    sf.write(save_path, data, sr, format="WAV")
                
                # Commit the volume to persist the speaker file
                speaker_volume.commit()
                return save_path
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process speaker audio: {e}"
                )
        
        return None
    
    @modal.fastapi_endpoint(method="GET")
    def root(self):
        """Root endpoint with API information"""
        return {
            "status": "ok",
            "service": "XTTS v2 Inference API on Modal",
            "model": "Genarabia-ai/Kuwaiti_XTTS_Latest",
            "endpoints": ["/healthz", "/tts", "/register_speaker"],
            "note": "Use /tts (POST) for synthesis. Use /register_speaker (POST) to register speakers.",
        }
    
    @modal.fastapi_endpoint(method="GET")
    def healthz(self):
        """Health check endpoint"""
        return PlainTextResponse("ok")
    
    @modal.fastapi_endpoint(method="POST")
    async def register_speaker(
        self, 
        req: RegisterSpeakerRequest,
        x_api_key: Optional[str] = None
    ):
        """Register a new speaker and return speaker_id"""
        self._require_api_key(x_api_key)
        
        if not req.speaker_wav_url and not req.speaker_wav_base64:
            raise HTTPException(
                status_code=400,
                detail="Either 'speaker_wav_url' or 'speaker_wav_base64' must be provided.",
            )
        
        # Get audio data
        audio_bytes = await self._get_audio_bytes(req)
        speaker_id = hashlib.sha1(audio_bytes).hexdigest()[:16]
        save_path = f"/speakers/{speaker_id}.wav"
        
        try:
            # Process and save audio
            with io.BytesIO(audio_bytes) as buf:
                data, sr = sf.read(buf, dtype="float32", always_2d=False)
                sf.write(save_path, data, sr, format="WAV")
            
            # Commit the volume to persist the speaker file
            speaker_volume.commit()
            
            return RegisterSpeakerResponse(speaker_id=speaker_id, path=save_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")
    
    @modal.fastapi_endpoint(method="POST")
    async def tts(
        self, 
        req: TTSRequest,
        x_api_key: Optional[str] = None
    ):
        """Generate speech from text"""
        self._require_api_key(x_api_key)
        
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        # Resolve speaker audio path
        speaker_path = await self._resolve_speaker_path(req)
        
        try:
            # Get conditioning latents from speaker audio
            gpt_cond_latent, speaker_embedding = (
                self.model.get_conditioning_latents(audio_path=[speaker_path])
                if speaker_path
                else (None, None)
            )
            
            # Generate speech
            with torch.no_grad():
                out = self.model.inference(
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


# Create a simple function for testing without the full class
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def simple_health():
    """Simple health check for testing"""
    return {"status": "ok", "message": "XTTS Modal server is running"}


if __name__ == "__main__":
    # This allows running the script locally for testing
    print("XTTS Modal server ready for deployment!")
    print("Deploy with: modal deploy modal_xtts_server_direct.py")
    print("Serve locally with: modal serve modal_xtts_server_direct.py")
