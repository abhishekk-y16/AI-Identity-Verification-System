import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class BehavioralService:
    """Behavioral biometrics — analyze typing, mouse, and interaction patterns."""

    def __init__(self):
        # Store user behavioral profiles (in production, use DB)
        self.profiles: dict[str, dict] = {}

    def extract_features(self, events: list[dict]) -> np.ndarray:
        """Extract a feature vector from behavioral events.
        
        Features: keystroke timing stats, mouse velocity stats, scroll patterns.
        """
        features = np.zeros(32, dtype=np.float32)

        keystroke_intervals = []
        mouse_velocities = []
        scroll_amounts = []

        sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))

        prev_keystroke_time = None
        prev_mouse_pos = None
        prev_mouse_time = None

        for event in sorted_events:
            etype = event.get("event_type", "")
            ts = event.get("timestamp", 0)
            data = event.get("data", {})

            if etype == "keystroke":
                if prev_keystroke_time is not None:
                    interval = ts - prev_keystroke_time
                    if 0 < interval < 2.0:  # Reasonable typing interval
                        keystroke_intervals.append(interval)
                prev_keystroke_time = ts

            elif etype == "mouse_move":
                x, y = data.get("x", 0), data.get("y", 0)
                if prev_mouse_pos is not None and prev_mouse_time is not None:
                    dt = ts - prev_mouse_time
                    if dt > 0:
                        dx = x - prev_mouse_pos[0]
                        dy = y - prev_mouse_pos[1]
                        velocity = np.sqrt(dx**2 + dy**2) / dt
                        mouse_velocities.append(velocity)
                prev_mouse_pos = (x, y)
                prev_mouse_time = ts

            elif etype == "scroll":
                amount = abs(data.get("delta", 0))
                scroll_amounts.append(amount)

        # Keystroke features (0-7)
        if keystroke_intervals:
            arr = np.array(keystroke_intervals)
            features[0] = np.mean(arr)
            features[1] = np.std(arr)
            features[2] = np.median(arr)
            features[3] = np.min(arr)
            features[4] = np.max(arr)
            features[5] = len(arr)
            features[6] = np.percentile(arr, 25)
            features[7] = np.percentile(arr, 75)

        # Mouse velocity features (8-15)
        if mouse_velocities:
            arr = np.array(mouse_velocities)
            features[8] = np.mean(arr)
            features[9] = np.std(arr)
            features[10] = np.median(arr)
            features[11] = np.min(arr)
            features[12] = np.max(arr)
            features[13] = len(arr)
            features[14] = np.percentile(arr, 25)
            features[15] = np.percentile(arr, 75)

        # Scroll features (16-23)
        if scroll_amounts:
            arr = np.array(scroll_amounts)
            features[16] = np.mean(arr)
            features[17] = np.std(arr)
            features[18] = np.median(arr)
            features[19] = len(arr)

        # General timing features (24-31)
        if sorted_events:
            timestamps = [e.get("timestamp", 0) for e in sorted_events]
            session_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            features[24] = session_duration
            features[25] = len(sorted_events)
            features[26] = len(sorted_events) / (session_duration + 1e-10)  # Event rate

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def update_profile(self, user_id: str, features: np.ndarray):
        """Update user's behavioral profile with new session data."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "feature_history": [],
                "mean_features": features.copy(),
                "count": 0,
            }

        profile = self.profiles[user_id]
        profile["feature_history"].append(features)
        # Keep last 20 sessions
        profile["feature_history"] = profile["feature_history"][-20:]
        # Running average
        profile["mean_features"] = np.mean(profile["feature_history"], axis=0)
        profile["count"] += 1

    def compare_behavior(self, user_id: str, features: np.ndarray) -> dict:
        """Compare current session behavior with user's profile."""
        if user_id not in self.profiles or self.profiles[user_id]["count"] < 2:
            return {
                "behavior_match_score": 0.5,
                "anomalies": [],
                "message": "Insufficient behavioral data for comparison — building profile",
            }

        profile = self.profiles[user_id]
        stored = profile["mean_features"]

        # Cosine similarity
        dot = np.dot(features, stored)
        norm1 = np.linalg.norm(features)
        norm2 = np.linalg.norm(stored)
        similarity = float(dot / (norm1 * norm2 + 1e-10))

        # Detect specific anomalies
        anomalies = []
        diff = np.abs(features - stored)
        high_deviation = np.where(diff > np.std(diff) * 2.0)[0]

        feature_names = {
            0: "typing_speed", 1: "typing_consistency", 8: "mouse_speed",
            9: "mouse_consistency", 16: "scroll_behavior", 25: "interaction_frequency",
        }
        for idx in high_deviation:
            name = feature_names.get(idx, f"feature_{idx}")
            anomalies.append(f"Unusual {name}")

        return {
            "behavior_match_score": max(0.0, similarity),
            "anomalies": anomalies[:5],
            "message": "Behavior matches profile" if similarity > 0.7 else "Behavioral anomaly detected",
        }

    async def analyze(self, user_id: str, events: list[dict]) -> dict:
        """Full behavioral analysis pipeline."""
        features = self.extract_features(events)
        result = self.compare_behavior(user_id, features)
        self.update_profile(user_id, features)
        return result
