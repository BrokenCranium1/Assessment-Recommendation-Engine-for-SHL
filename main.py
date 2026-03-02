from fastapi import FastAPI
import os
import logging

# Minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "message": "Minimal app is working!"}

@app.get("/")
async def root():
    return {"message": "SHL Recommendation API - Use /health or /recommend"}

@app.post("/recommend")
async def recommend(query: str = "java developer"):
    # This is a dummy response to test the endpoint
    return [
        {
            "url": "https://example.com/test",
            "name": "Test Assessment",
            "adaptive_support": "No",
            "description": "This is a test response",
            "duration": 30,
            "remote_support": "Yes",
            "test_type": ["K"]
        }
    ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting minimal app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)