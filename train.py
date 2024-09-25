

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
bank_d = pd.read_csv('/mnt/datalake/iota/BankChurn.csv')

# Split the data into features (X) and target (y)
X = bank_d.drop('churn', axis=1)  # Assuming 'churn' is your target variable
y = bank_d['churn']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline:
# 1. OneHotEncode the categorical columns
# 2. Use SimpleImputer to handle missing values in numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),  # Encode categorical columns
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(exclude=['object']).columns)  # Impute missing values in numeric columns
    ])

# Create a pipeline with the preprocessor and the LinearRegression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Extract the trained model from the pipeline
model = pipeline.named_steps['model']

# Predict using the test set
y_pred = model.predict(pipeline.named_steps['preprocessor'].transform(X_test))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save only the model
joblib.dump(model, '/mnt/datalake/iota/bankchurn_model.pkl')

