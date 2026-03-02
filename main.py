from fastapi import FastAPI
import os
import logging
from engine import RecommendationEngine  # Add this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Try loading engine at startup
try:
    logger.info("Attempting to load engine...")
    engine = RecommendationEngine("data/shl_catalog_final.csv")
    logger.info("✅ Engine loaded successfully!")
except Exception as e:
    logger.error(f"❌ Engine failed to load: {e}")
    engine = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}

@app.get("/")
async def root():
    return {"message": "SHL Recommendation API"}

# Keep your working minimal endpoint for now
@app.post("/recommend")
async def recommend(query: str = "java developer"):
    if engine is None:
        return {"error": "Engine not loaded"}
    
    try:
        # Try using the real engine
        results = engine.get_balanced_recommendations(query, top_k=3)
        return results
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        # Fallback to dummy data
        return [
            {
                "url": "https://example.com/test",
                "name": f"Test for: {query}",
                "adaptive_support": "No",
                "description": "Fallback response",
                "duration": 30,
                "remote_support": "Yes",
                "test_type": ["K"]
            }
        ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)