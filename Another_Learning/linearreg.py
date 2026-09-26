# Import required library
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: X = house size (sqft), y = price
X = np.array([[500], [1000], [1500], [2000]])   # Independent variable
y = np.array([4700000, 6500000, 7500000, 14000000])  # Dependent variable
z = np.array([6000000, 7800000, 8700000, 20000000])
# Create a Linear Regression model
model = LinearRegression()

# Train (fit) the model
model.fit(X, y, z)

# Predict price for a new house of 1200 sqft
pred = model.predict([[1860]])

print("🏠 Predicted Price for 1860 sqft:", pred[0])