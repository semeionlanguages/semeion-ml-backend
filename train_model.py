import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from preprocessing import encode_metadata

# === Simulated enriched training data ===
# Each entry: [times_seen, time_since_last_review, recall_score] + metadata
def make_sample(times_seen, time_since, recall_score, register, mtype, pos, gender, freq):
    base = [times_seen, time_since, recall_score]
    meta = encode_metadata(register, mtype, pos, gender, freq)
    return base + meta

X = np.array([
    make_sample(1, 86400, 0, "formal", "literal", "noun", "masculine", "common"),
    make_sample(2, 43200, 1, "slang", "idiomatic", "verb", "feminine", "rare"),
    make_sample(3, 604800, 1, "neutral", "literal", "noun", "masculine", "extremely common"),
    make_sample(5, 300, 0, "technical", "compound", "noun", "neuter", "rare"),
    make_sample(10, 864000, 1, "poetic", "archaism", "adjective", "feminine", "extremely rare"),
    make_sample(7, 7200, 0, "unknown", "unknown", "unknown", "unknown", "unknown")  # 👈 NEW!
])


# Corresponding labels: did the user recall the word?
y = np.array([0, 1, 1, 0, 1, 1])

# Train and save model
model = RandomForestClassifier()
model.fit(X, y)

with open("recall_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Enhanced model trained and saved.")
