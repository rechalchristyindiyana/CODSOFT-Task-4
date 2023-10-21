# CODSOFT-Task-4
Task-4
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
data = pd.read_csv("/content/advertising.csv")
data
#We extract the "TV," "Radio," and "Newspaper" columns as features and the "Sales" column as the target variable.
# Extract features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
X
y = data['Sales']
y
# Split the dataset into training and testing sets(80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#A Linear Regression model is initialized and trained on the training data
# Initialize and train a Linear Regression model
model = LinearRegression()
model
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred
#We make predictions on the test data and evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R2) score.
# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)
# Now, you can use the model to make sales predictions for new data
new_data = pd.DataFrame({'TV': [100],
                         'Radio': [20],
                         'Newspaper': [10]})
new_data
#Finally, we can use the trained model to make sales predictions for new data. In this example, we provided new data for prediction
#sales predictions for new data
predicted_sales = model.predict(new_data)
print("Predicted Sales:", predicted_sales[0])
