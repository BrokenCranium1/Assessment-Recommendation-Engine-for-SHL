from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from engine import RecommendationEngine
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommendation API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
CATALOG_PATH = "data/shl_catalog_final.csv"
if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Catalog file {CATALOG_PATH} not found!")

engine = RecommendationEngine(CATALOG_PATH)

# Models
class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    logger.info(f"Received recommendation request: {request.query}, top_k: {request.top_k}")
    try:
        # Get recommendations from engine
        results = engine.get_balanced_recommendations(request.query, top_k=request.top_k)
        
        # Format results with EXACT field names from PDF Appendix 2
        formatted_results = []
        for r in results:
            # Clean up test_type string to list
            test_types = []
            if r.get('test_type'):
                if isinstance(r['test_type'], list):
                    test_types = r['test_type']
                else:
                    test_types = [t.strip() for t in str(r['test_type']).split(',')]
            
            # Ensure duration is integer
            try:
                duration_val = r.get('duration')
                if duration_val is None or str(duration_val).strip() == '':
                    duration = 0
                else:
                    duration = int(float(duration_val))
            except (ValueError, TypeError):
                duration = 0
                
            formatted_results.append(AssessmentResponse(
                url=r.get('url', ''),
                name=r.get('name', ''),
                adaptive_support=r.get('adaptive_support', 'No'),
                description=r.get('description', ''),
                duration=duration,
                remote_support=r.get('remote_support', 'No'),
                test_type=test_types
            ))
            
        return RecommendResponse(recommended_assessments=formatted_results)
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)