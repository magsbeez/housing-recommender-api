import uvicorn
from fastapi import FastAPI
from typing import List, Dict, Any
import numpy as np

# Import the classes from your finished invention file
from housing_recommender import ReinforcementEngine, ListingNode, ExternalDataSource

# --- 1. INITIALIZE THE AI ENGINE ---
# This creates one instance of your AI engine when the server starts.
print("--- [SERVER] INITIALIZING SMART HOUSING RECOMMENDER ---")
Engine = ReinforcementEngine(university_focus="University of Connecticut")

# --- 2. PRE-LOAD ENGINE WITH DATA (SIMULATION) ---
# In a real app, this data would come from the API client.
Engine.ingest_data(
    ListingNode(price=1800, distance_to_campus=500, amenities_score=0.9, description="Luxury near campus."))
Engine.ingest_data(
    ListingNode(price=1200, distance_to_campus=4000, amenities_score=0.5, description="Cheap, far apartment."))
Engine.ingest_data(ListingNode(price=1500, distance_to_campus=1500, amenities_score=0.8,
                               description="Standard middle-tier apartment."))
Engine.ingest_data(
    ListingNode(price=1450, distance_to_campus=1000, amenities_score=0.7, description="Good value apartment."))
print(f"--- [SERVER] AI Engine loaded with {len(Engine.listings)} initial listings. ---")

# --- 3. DEFINE THE API ENDPOINTS ---
app = FastAPI(
    title="Smart Housing Recommender API",
    description="AI-powered recommendations using Reinforcement Learning.",
    version="1.0.0"
)


@app.get("/recommendations/")
async def get_recommendations(budget: float, distance_km: float) -> List[Dict[str, Any]]:
    """
    Runs the RL Engine to get the top recommendations based on user constraints.
    """
    # Use the 'generate_recommendation' method from your invention
    recommendations = Engine.generate_recommendation(budget, distance_km)

    # Format the output as JSON for the website/app
    output = [
        {
            "listing_id": rec.listing_id,
            "description": rec.description,
            "price": rec.price,
            "distance_km": rec.distance / 1000,
            "predicted_value": rec.rl_prediction_value
        } for rec in recommendations
    ]
    return output


@app.post("/feedback/")
async def log_user_feedback(listing_id: str, feedback_type: str):
    """
    Receives user feedback (e.g., 'CLICKED_SAVE') and logs it
    to the Replay Buffer for future AI retraining.
    """
    # Simulate a generic state for the RL buffer (in a real app, this would be more complex)
    sim_state = np.array([1500, 2000, 0.7])

    Engine.process_user_feedback(
        listing_id=listing_id,
        feedback_type=feedback_type,
        current_state=sim_state,
        next_state=sim_state
    )

    return {"status": "success", "message": f"Feedback '{feedback_type}' logged for {listing_id}."}


# --- 4. RUN THE SERVER (FOR LOCAL TESTING) ---
if __name__ == "__main__":
    # This command makes the server run on http://127.0.0.1:8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)