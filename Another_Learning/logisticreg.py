# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Training data: hours studied (X) vs pass/fail (y)
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]  # 0 = Fail, 1 = Pass

# Create model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict for 3.5 hours of study
prediction = model.predict([[8]])
probability = model.predict_proba([[8]])

print("🎓 Prediction (1=Pass, 0=Fail):", prediction[0])
print("🔢 Probability of Passing:", probability[0][1])
