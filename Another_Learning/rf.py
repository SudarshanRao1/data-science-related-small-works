# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Same dataset (AND logic)
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 0, 0, 1]

# Create Random Forest with multiple trees
forest = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the model
forest.fit(X, y)

# Predict for new input
print("🌳 Random Forest Prediction for [1,1]:", forest.predict([[1,1]])[0])
