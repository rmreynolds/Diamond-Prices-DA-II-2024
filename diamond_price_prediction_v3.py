import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

# Read the CSV files
df_raw = pd.read_csv("Sarah Gets a Diamond raw data.csv.csv")
df_out_of_sample = pd.read_csv("Sarah Gets a Diamond out of sample prices.csv")

# Filter the first 6000 rows for training while preserving the 'Price' column
df_train = df_raw.iloc[:6000].dropna(subset=['Price']).copy()

# Combine remaining rows with out-of-sample data for testing, dropping the 'Price' column if it exists in the out-of-sample data.
df_test = pd.concat([df_raw.iloc[6000:].drop(columns=['Price'], errors='ignore'), df_out_of_sample], ignore_index=True)

# Identify categorical and numerical columns
categorical_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
numerical_col = ['Carat Weight']

# Impute missing values in categorical columns
imputation_values = df_train[categorical_cols].mode().iloc[0]
df_train[categorical_cols] = df_train[categorical_cols].fillna(imputation_values)
df_test[categorical_cols] = df_test[categorical_cols].fillna(imputation_values)

# Impute missing values in the numerical column
carat_weight_mean = df_train[numerical_col].mean()
df_test[numerical_col] = df_test[numerical_col].fillna(carat_weight_mean)

# One-hot encode categorical columns without dropping the first category
encoder = OneHotEncoder()  
X_train_cat = encoder.fit_transform(df_train[categorical_cols]).toarray()
X_test_cat = encoder.transform(df_test[categorical_cols]).toarray()

# Create DataFrames for encoded features
encoded_train_df = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(categorical_cols))
encoded_test_df = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded features with numerical features
X_train = pd.concat([df_train[numerical_col], encoded_train_df], axis=1)
X_test = pd.concat([df_test[numerical_col], encoded_test_df], axis=1)

# Create interaction terms between 'Carat Weight' and one-hot encoded features
for col in encoded_train_df.columns:
    X_train[f'Carat Weight_{col}'] = X_train['Carat Weight'] * X_train[col]
    X_test[f'Carat Weight_{col}'] = X_test['Carat Weight'] * X_test[col]

# Extract target variable
y_train = df_train['Price']

# Remove infinite values and values too large for float64 from y_train
y_train = y_train[np.isfinite(y_train)]
y_train = y_train.clip(upper=np.finfo(np.float64).max)

# Split the training data into a training set and a validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train_split, y_train_split)

# Predict on the validation set and calculate MAPE
y_pred_lr = model_lr.predict(X_val)
mape_lr = mean_absolute_percentage_error(y_val, y_pred_lr)
print(f"Linear Regression MAPE: {mape_lr:.2%}")

# Train a Random Forest model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train_split, y_train_split)

# Predict on the validation set and calculate MAPE
y_pred_rf = model_rf.predict(X_val)
mape_rf = mean_absolute_percentage_error(y_val, y_pred_rf)
print(f"Random Forest MAPE: {mape_rf:.2%}")

# Select the best model based on MAPE
best_model = model_rf if mape_rf < mape_lr else model_lr

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Create DataFrame with predictions for the ENTIRE test set
df_test_predictions = pd.DataFrame({'ID': df_test['ID'], 'Predicted_Price': y_pred})

# Save predictions to a new CSV file
df_test_predictions.to_csv("predicted_prices_updated.csv", index=False)