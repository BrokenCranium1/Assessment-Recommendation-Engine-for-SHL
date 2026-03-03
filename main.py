from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ✅ ADD CORS MIDDLEWARE - This fixes the 405 OPTIONS errors!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows ALL methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# Global variable to store catalog path
catalog_path = None
engine = None

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
    found_path = find_catalog_file()
    
    if found_path is None:
        raise FileNotFoundError("Could not find catalog file in any expected location")
    
    # Save to global variable
    catalog_path = found_path
    
    engine = RecommendationEngine(found_path)
    logger.info("✅ Engine loaded successfully!")
except Exception as e:
    logger.error(f"❌ Engine failed to load: {e}")
    engine = None
    catalog_path = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "engine_loaded": engine is not None,
        "catalog_found": catalog_path
    }

@app.get("/")
async def root():
    return {"message": "SHL Recommendation API"}

# ✅ Handle OPTIONS requests explicitly (though CORS middleware should handle it)
@app.options("/recommend")
async def options_recommend():
    return {"message": "OK"}

@app.post("/recommend")
async def recommend(query: str = "java developer"):
    if engine is None:
        return {"error": "Engine not loaded", "catalog_path": catalog_path}
    
    try:
        # Get raw results from engine
        raw_results = engine.get_balanced_recommendations(query, top_k=5)
        
        # Format results to match PDF Appendix 2 EXACTLY
        formatted_results = []
        for r in raw_results:
            # Handle test_type - ensure it's always a list
            test_type = r.get("test_type", "K")
            if isinstance(test_type, str):
                # If it's a comma-separated string, split it
                if "," in test_type:
                    test_type_list = [t.strip() for t in test_type.split(",")]
                else:
                    test_type_list = [test_type]
            elif isinstance(test_type, list):
                test_type_list = test_type
            else:
                test_type_list = ["K"]
            
            # Ensure duration is integer
            try:
                duration_val = r.get("duration", 0)
                if duration_val is None:
                    duration = 0
                else:
                    duration = int(float(duration_val))
            except (ValueError, TypeError):
                duration = 0
            
            # Create clean response object with ONLY the 7 required fields
            formatted_results.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "adaptive_support": r.get("adaptive_support", "No"),
                "description": r.get("description", ""),
                "duration": duration,
                "remote_support": r.get("remote_support", "No"),
                "test_type": test_type_list
            })
        
        return formatted_results[:5]  # Ensure max 5 results
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        # Fallback to dummy data that matches the required format
        return [
            {
                "url": "https://example.com/test",
                "name": f"Test for: {query}",
                "adaptive_support": "No",
                "description": f"Fallback response - engine error: {str(e)}",
                "duration": 30,
                "remote_support": "Yes",
                "test_type": ["K"]
            }
        ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)