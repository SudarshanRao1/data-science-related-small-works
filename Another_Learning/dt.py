# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Simple dataset (AND logic)
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 0, 0, 1]

# Create model
tree = DecisionTreeClassifier(criterion='gini')

# Train the model
tree.fit(X, y)

# Predict for new data
print("🌲 Prediction for [1,1]:", tree.predict([[1,1]])[0])
