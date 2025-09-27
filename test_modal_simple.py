import modal

# Simple test image
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pydantic"
)

# Create test app
app = modal.App("xtts-server-test", image=image)

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "XTTS Modal server test is running"}

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def info():
    """Info endpoint"""
    return {
        "service": "XTTS Modal Server Test",
        "version": "1.0.0",
        "endpoints": ["/health", "/info"]
    }

if __name__ == "__main__":
    print("Simple Modal test server ready!")
    print("Deploy with: modal deploy test_modal_simple.py")
    print("Serve with: modal serve test_modal_simple.py")
