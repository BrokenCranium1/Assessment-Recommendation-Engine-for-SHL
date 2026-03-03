from fastapi import FastAPI
import os
import logging
from engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Try multiple possible paths for the catalog file
def find_catalog_file():
    possible_paths = [
        "data/shl_catalog_final.csv",           # Local development
        "/app/data/shl_catalog_final.csv",       # Docker default
        "./data/shl_catalog_final.csv",          # Relative with dot
        "../data/shl_catalog_final.csv",         # Parent directory
        os.path.join(os.path.dirname(__file__), "data", "shl_catalog_final.csv"),  # Same dir as script
        os.path.join("/app", "data", "shl_catalog_final.csv"),  # Explicit /app path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found catalog at: {path}")
            return path
    
    # If not found, list directory contents for debugging
    logger.error("❌ Catalog file not found in any expected location")
    try:
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('data'):
            logger.info(f"Files in ./data: {os.listdir('data')}")
        if os.path.exists('/app'):
            logger.info(f"Files in /app: {os.listdir('/app')}")
        if os.path.exists('/app/data'):
            logger.info(f"Files in /app/data: {os.listdir('/app/data')}")
    except Exception as e:
        logger.error(f"Error listing directories: {e}")
    
    return None

# Try loading engine at startup
try:
    logger.info("Attempting to load engine...")
    catalog_path = find_catalog_file()
    
    if catalog_path is None:
        raise FileNotFoundError("Could not find catalog file in any expected location")
    
    engine = RecommendationEngine(catalog_path)
    logger.info("✅ Engine loaded successfully!")
except Exception as e:
    logger.error(f"❌ Engine failed to load: {e}")
    engine = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "engine_loaded": engine is not None,
        "catalog_found": catalog_path if 'catalog_path' in locals() else None
    }

@app.get("/")
async def root():
    return {"message": "SHL Recommendation API"}

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