import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.array([
    [1, 86400, 0],
    [2, 43200, 1],
    [3, 604800, 1],
    [5, 300, 0],
    [10, 864000, 1],
])
y = np.array([0, 1, 1, 0, 1])

model = RandomForestClassifier()
model.fit(X, y)

with open("recall_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")
