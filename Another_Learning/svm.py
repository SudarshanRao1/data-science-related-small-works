# Import SVM
from sklearn import svm

# Training data
X = [[1,2], [2,3], [3,3], [8,8], [9,9], [10,10]]
y = [0, 0, 0, 1, 1, 1]  # Two classes

# Create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# Train the model
model.fit(X, y)

# Predict for new point
print("📈 Prediction for [4,4]:", model.predict([[4,4]])[0])
