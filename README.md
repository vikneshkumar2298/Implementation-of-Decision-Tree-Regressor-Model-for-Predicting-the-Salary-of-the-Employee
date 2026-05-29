# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 

1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:

```
# Import libraries

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the data

df = pd.read_csv('Salary.csv')

# Display the data

print("Salary Data:")

print(df)
print()

# Prepare data

X = df[['Level']]  # Feature (Level)
y = df['Salary']    # Target (Salary)

# Create and train the model

model = DecisionTreeRegressor()
model.fit(X, y)

# Make predictions for all levels

predictions = model.predict(X)

# Show predictions

print("Actual vs Predicted Salaries:")
for i in range(len(df)):
    print(f"Level {df.iloc[i]['Level']}: Actual=${df.iloc[i]['Salary']}, Predicted=${int(predictions[i])}")

# Calculate accuracy (R² score)

accuracy = model.score(X, y)
print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")

# Predict salary for a new level

new_level = [[6.5]]
predicted_salary = model.predict(new_level)
print(f"\nPredicted Salary for Level 6.5: ${int(predicted_salary[0])}")

# Simple visualization

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=100)
plt.plot(X, predictions, color='red', label='Decision Tree Predictions', linewidth=2)
plt.xlabel('Level')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction using Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:

Salary Data :


<img width="384" height="263" alt="image" src="https://github.com/user-attachments/assets/209dda24-e0d8-4501-81d4-642eb8c3c739" />

Actual vs Predicted :


<img width="452" height="238" alt="image" src="https://github.com/user-attachments/assets/b924c9e5-e8b2-46b2-942e-378ad0ac01ed" />

Accuracy :


<img width="389" height="66" alt="image" src="https://github.com/user-attachments/assets/61805a1a-b910-43de-afde-e11a37e19358" />

Decision Tree :


<img width="885" height="593" alt="image" src="https://github.com/user-attachments/assets/1199d04b-21b4-4056-bfa8-06ce43613e75" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
