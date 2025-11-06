import uuid
from datetime import datetime
from typing import Dict, Any, List
from collections import deque
import random
import numpy as np
from math import sqrt


# --- CORE DATA MODEL ---

class ListingNode:
    """
    Represents a single apartment or house listing, comparable to a CausalNode in value.
    """

    def __init__(self, price: float, distance_to_campus: float, amenities_score: float, description: str):
        self.listing_id = str(uuid.uuid4())
        self.price = price
        self.distance = distance_to_campus  # in meters
        self.amenities = amenities_score  # 0.0 to 1.0
        self.description = description
        self.time_listed = datetime.now()

        # RL Engine Inputs (These will be learned and updated)
        self.user_satisfaction_score = 0.0  # Feedback from the user (reward signal)
        self.rl_prediction_value = 0.0  # The engine's predicted value of this listing

    def __repr__(self):
        return f"Listing(Price=${self.price:.2f}, Dist={self.distance / 1000:.1f}km, Sat={self.user_satisfaction_score:.2f})"


# --- ADVANCED RL MECHANISMS ---

class ReplayBuffer:
    """
    Stores past experiences (state, action, reward, next_state) to break
    temporal correlation and stabilize the DQN training process.
    """

    def __init__(self, buffer_size: int = 50000):
        # deque is highly efficient for managing a fixed-size buffer
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, current_state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Adds a single transition of user interaction to the memory."""
        experience = (current_state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple:
        """Randomly samples a batch of past experiences for DQN training."""
        if len(self.buffer) < batch_size:
            return None

        # Randomly select a batch of experiences from the buffer
        sample_batch = random.sample(self.buffer, batch_size)

        # 'zip' is used to separate the components back into separate arrays
        states, actions, rewards, next_states, dones = zip(*sample_batch)

        # Convert to numpy arrays for the simulated neural network input
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        """Returns the current number of experiences stored."""
        return len(self.buffer)


class ReinforcementEngine:
    """
    Simulates the Deep Q-Network (DQN) that learns optimal housing recommendations.
    This is the "brain" of the recommender.
    """

    def __init__(self, university_focus: str, buffer_size: int = 50000):
        self.focus = university_focus
        self.listings: List[ListingNode] = []
        self.q_network_features = 47  # Based on established deep Q-Network models for recommendations
        self.replay_buffer = ReplayBuffer(buffer_size)

    def ingest_data(self, new_listing: ListingNode):
        """Adds a new data point to the recommendation engine."""
        self.listings.append(new_listing)

    def generate_recommendation(self, user_budget: float, max_distance_km: float) -> List[ListingNode]:
        """
        Simulates the RL algorithm prioritizing listings that maximize satisfaction
        within user constraints.
        """
        print(f"\n[RL Engine] Running {self.q_network_features} feature analysis for {self.focus}...")

        eligible = [
            l for l in self.listings
            if l.price <= user_budget and l.distance <= max_distance_km * 1000
        ]

        # Simplified RL sorting: Prioritize by predicted value (which would be learned by the Q-Network)
        # We simulate a learned preference for closer and cheaper listings:
        for listing in eligible:
            listing.rl_prediction_value = (user_budget - listing.price) + (5000 - listing.distance) + (
                        listing.amenities * 1000)

        # Sort by the predicted value (highest first)
        eligible.sort(key=lambda x: x.rl_prediction_value, reverse=True)

        return eligible[:3]  # Return Top 3 recommendations

    def process_user_feedback(self, listing_id: str, feedback_type: str, current_state: np.ndarray,
                              next_state: np.ndarray):
        """
        Processes user actions (the reward signal), updates the listing, and stores the experience.
        """
        for i, listing in enumerate(self.listings):
            if listing.listing_id == listing_id:
                reward = 0.0
                done = False

                if feedback_type == "LEASE_SIGNED":
                    reward = 10.0  # High positive reward
                    done = True  # Interaction cycle completed
                elif feedback_type == "CLICKED_SAVE":
                    reward = 1.0  # Medium positive reward
                elif feedback_type == "SKIPPED_ITEM":
                    reward = -0.5  # Negative reward

                # Update the listing's satisfaction score
                listing.user_satisfaction_score += reward
                print(f"[RL Reward] Listing {listing_id[:4]} received reward: {reward:.1f}")

                # Store the experience in the Replay Buffer
                # The action is the index of the listing that was recommended (simulated)
                self.replay_buffer.add_experience(current_state, action=i, reward=reward, next_state=next_state,
                                                  done=done)
                print(f"[Buffer] Experience logged. Buffer Size: {self.replay_buffer.size()}")
                return

    def run_retraining_simulation(self):
        """
        Simulates the periodic retraining of the Deep Q-Network (DQN) policy
        by sampling a batch of data from the Replay Buffer.
        """
        print(f"\n[RL Training] Initiating Deep Q-Network (DQN) retraining...")
        total_rewards = sum(l.user_satisfaction_score for l in self.listings)

        BATCH_SIZE = 4  # We need at least 4 experiences for a test batch

        if self.replay_buffer.size() >= BATCH_SIZE:
            training_batch = self.replay_buffer.sample(BATCH_SIZE)
            sampled_rewards = training_batch[2]  # Rewards are the third element in the tuple

            # Simulate the DQN updating its policy to maximize the total reward
            print(f"  Policy updated. Total cumulative reward (RL Objective): {total_rewards:.2f}")
            print(f"  DQN sampled rewards: {sampled_rewards}. Training policy now optimizing.")
        else:
            print(f"  Insufficient data (Need {BATCH_SIZE} experiences) for deep learning model retraining.")


# --- EXTERNAL API INTEGRATION ---

class ExternalDataSource:
    """
    Simulates connecting to an external real estate API (e.g., Zillow/Trulia)
    to retrieve new, live listing data for the RL Engine.
    """

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        print(f"[API] Initializing connection to {self.endpoint}...")

    def fetch_new_listings(self, count: int) -> List[Dict[str, Any]]:
        """
        Simulates an API call that retrieves structured listing data.
        In reality, this involves HTTPS requests, header management, and JSON parsing.
        """
        if not self.api_key or "demo" in self.api_key.lower():
            print("[API] WARNING: Using simulated data due to invalid API key.")

        # Simulated live data generation:
        listings = []
        for i in range(count):
            listings.append({
                "price": round(1000 + (i * 100) + (random.randint(0, 100)), 2),
                "distance": round(500 + (i * 500) + (random.randint(0, 200)), 2),
                "amenities_score": round(random.uniform(0.4, 1.0), 2),
                "description": f"New Listing {uuid.uuid4().hex[:4]} - High Floor"
            })

        return listings

    def transform_and_ingest(self, raw_data: List[Dict[str, Any]], engine: 'ReinforcementEngine'):
        """
        Transforms raw API data into the ListingNode format required by the RL Engine
        and ingests it for learning.
        """
        for data in raw_data:
            new_node = ListingNode(
                price=data['price'],
                distance_to_campus=data['distance'],
                amenities_score=data['amenities_score'],
                description=data['description']
            )
            engine.ingest_data(new_node)
        print(f"[API Ingest] Successfully ingested {len(raw_data)} new listings into the RL Engine.")


# --- INITIALIZATION AND TEST SEQUENCE (MAIN EXECUTION) ---
if __name__ == "__main__":
    print("--- PROJECT: SMART HOUSING RECOMMENDER INITIALIZING ---")

    # 1. Initialize the RL Engine
    Engine = ReinforcementEngine(university_focus="University of Connecticut")

    # 2. Ingest Sample Data (Input for the RL Agent)
    Engine.ingest_data(ListingNode(price=1800, distance_to_campus=500, amenities_score=0.9,
                                   description="Luxury near campus."))  # Index 0
    Engine.ingest_data(ListingNode(price=1200, distance_to_campus=4000, amenities_score=0.5,
                                   description="Cheap, far apartment."))  # Index 1
    Engine.ingest_data(ListingNode(price=1500, distance_to_campus=1500, amenities_score=0.8,
                                   description="Standard middle-tier apartment."))  # Index 2
    Engine.ingest_data(ListingNode(price=1450, distance_to_campus=1000, amenities_score=0.7,
                                   description="Good value apartment."))  # Index 3

    # 3. Define the User State (Input for the DQN)
    # State Vector: [Budget, Max Distance, Required Amenities Score]
    USER_STATE_1 = np.array([1600, 2000, 0.8])
    USER_NEXT_STATE = np.array([1600, 1500, 0.8])  # User slightly reduced their max distance after interaction

    # 4. Generate Initial Recommendations
    user_budget = 1600
    max_dist = 2.0  # km

    recommendations = Engine.generate_recommendation(user_budget, max_dist)

    print(f"\n[User Constraints] Budget: ${user_budget}, Max Distance: {max_dist}km")
    print("\n--- TOP 3 RL RECOMMENDATIONS (Policy 1) ---")

    for i, rec in enumerate(recommendations):
        print(
            f"{i + 1}. {rec.description}: Price=${rec.price}, Dist={rec.distance / 1000:.1f}km, Predicted Value={rec.rl_prediction_value:.0f}")

    # 5. Simulate User Interaction (The RL Feedback Loop)
    print("\n--- SIMULATING RL FEEDBACK LOOP & EXPERIENCE LOGGING ---")

    # Interaction 1: User likes the 'Good value apartment' (Index 3)
    Engine.process_user_feedback(
        listing_id=Engine.listings[3].listing_id,
        feedback_type="CLICKED_SAVE",
        current_state=USER_STATE_1,
        next_state=USER_NEXT_STATE
    )

    # Interaction 2: User skips the expensive 'Luxury' listing (Index 0)
    Engine.process_user_feedback(
        listing_id=Engine.listings[0].listing_id,
        feedback_type="SKIPPED_ITEM",
        current_state=USER_STATE_1,
        next_state=USER_NEXT_STATE
    )

    # Interaction 3: User signs the lease on the 'Standard' listing (Index 2)
    Engine.process_user_feedback(
        listing_id=Engine.listings[2].listing_id,
        feedback_type="LEASE_SIGNED",
        current_state=USER_STATE_1,
        next_state=USER_NEXT_STATE
    )

    # Interaction 4: User skips the 'Cheap, far' listing (Index 1)
    Engine.process_user_feedback(
        listing_id=Engine.listings[1].listing_id,
        feedback_type="SKIPPED_ITEM",
        current_state=USER_STATE_1,
        next_state=USER_NEXT_STATE
    )

    # 6. Run Retraining Simulation
    Engine.run_retraining_simulation()

    # 7. Test External API Integration
    print("\n" + "=" * 50)
    print("--- TESTING EXTERNAL API INTEGRATION ---")

    # Initialize the data source client
    ZILLOW_ENDPOINT = "https://api.zillow.com/listings"
    API_CLIENT = ExternalDataSource(api_key="DEMO_API_KEY_123", endpoint=ZILLOW_ENDPOINT)

    # Fetch data from the simulated API
    new_raw_listings = API_CLIENT.fetch_new_listings(count=5)

    # Transform and ingest data into the RL Engine
    API_CLIENT.transform_and_ingest(new_raw_listings, Engine)

    # 8. Verify Final State
    print("\n--- FINAL STATE VERIFICATION ---")
    print(f"Total Experiences Logged in Buffer: {Engine.replay_buffer.size()}")
    print(f"Final Cumulative Satisfaction (Reward): {Engine.listings[2].user_satisfaction_score:.2f}")
    print(f"Total Listings in RL Engine (Verification): {len(Engine.listings)}")