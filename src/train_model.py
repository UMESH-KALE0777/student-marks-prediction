import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import json

# Load dataset
df = pd.read_csv("data/student_marks.csv")

X = df[['Hours_Studied', 'Attendance', 'Assignments_Submitted']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Calculate accuracy (R² Score)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# Save model
pickle.dump(model, open("models/linear_model.pkl", "wb"))

# Save accuracy
with open("models/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("Model trained! Accuracy:", accuracy)
