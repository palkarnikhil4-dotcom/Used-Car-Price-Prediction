import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Car details v3.csv")

# Create a copy of the original dataset
data = df.copy()

# Display the first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Display dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Check dataset information
print("\nDATASET INFORMATION")
df.info()

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Remove duplicate records
data = data.drop_duplicates()

print("\nShape after removing duplicates:")
print(data.shape)

# Create car age
data["car_age"] = 2021 - data["year"]

print("\nCar age created successfully.")
print(data[["year", "car_age"]].head())

# Convert mileage to numerical value
data["mileage"] = data["mileage"].str.extract(r"([\d.]+)").astype(float)

# Convert engine to numerical value
data["engine"] = data["engine"].str.extract(r"([\d.]+)").astype(float)

# Convert max_power to numerical value
data["max_power"] = data["max_power"].str.extract(r"([\d.]+)").astype(float)

# Fill missing numerical values with median
numeric_columns = ["mileage", "engine", "max_power", "seats"]

for column in numeric_columns:
    data[column] = data[column].fillna(data[column].median())

   # Remove torque because it contains different units and formats
data = data.drop(columns=["torque"])

# Extract car brand from car name
data["brand"] = data["name"].str.split().str[0]

# Remove original name column
data = data.drop(columns=["name"])

print("\nCLEANED DATA")
print(data.head())

print("\nMissing values after preprocessing:")
print(data.isnull().sum())

print("\nFinal shape:")
print(data.shape)

print("\nFinal columns:")
print(data.columns.tolist())

# -----------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------

# Distribution of selling price
plt.figure(figsize=(10, 6))

sns.histplot(data["selling_price"], bins=50, kde=True)

plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (₹)")
plt.ylabel("Number of Cars")

plt.show()

# Selling price vs car age
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x="car_age",
    y="selling_price",
    data=data,
    alpha=0.5
)

plt.title("Selling Price vs Car Age")
plt.xlabel("Car Age (years)")
plt.ylabel("Selling Price (₹)")

plt.show()

# Distribution of car brands
plt.figure(figsize=(12, 6))

brand_counts = data["brand"].value_counts().head(15)

sns.barplot(
    x=brand_counts.values,
    y=brand_counts.index
)

plt.title("Top 15 Car Brands in the Dataset")
plt.xlabel("Number of Cars")
plt.ylabel("Car Brand")

plt.show()

# Average selling price by car brand
brand_price = data.groupby("brand")["selling_price"].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=brand_price.values,
    y=brand_price.index
)

plt.title("Average Selling Price by Car Brand")
plt.xlabel("Average Selling Price (₹)")
plt.ylabel("Car Brand")

plt.show()

# Average selling price by fuel type
fuel_price = data.groupby("fuel")["selling_price"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=fuel_price.index,
    y=fuel_price.values
)

plt.title("Average Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price (₹)")

plt.show()

# Average selling price by transmission type
transmission_price = data.groupby("transmission")["selling_price"].mean()

plt.figure(figsize=(8, 6))

sns.barplot(
    x=transmission_price.index,
    y=transmission_price.values
)

plt.title("Average Selling Price by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Average Selling Price (₹)")

plt.show()

# Selling price vs mileage
plt.figure(figsize=(10, 6))

plt.scatter(
    data["mileage"],
    data["selling_price"],
    alpha=0.5
)

plt.title("Selling Price vs Mileage")
plt.xlabel("Mileage (km/l)")
plt.ylabel("Selling Price (₹)")

plt.show()

# Selling price vs engine size
plt.figure(figsize=(10, 6))

plt.scatter(
    data["engine"],
    data["selling_price"],
    alpha=0.5
)

plt.title("Selling Price vs Engine Size")
plt.xlabel("Engine Size (CC)")
plt.ylabel("Selling Price (₹)")

plt.show()

# ==============================
# MACHINE LEARNING MODEL
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Separate input features and target
X = data.drop(columns=["selling_price"])
y = data["selling_price"]

# Categorical columns
categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand"
]

# Numerical columns
numerical_columns = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "car_age"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# Train the model
pipeline.fit(X_train, y_train)

print("\nModel trained successfully!")

# ==============================
# MODEL EVALUATION
# ==============================

# Make predictions on test data
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# ==============================
# CAR PRICE PREDICTION
# ==============================

print("\nENTER CAR DETAILS")

year = int(input("Enter car year: "))
km_driven = float(input("Enter kilometers driven: "))
fuel = input("Enter fuel type (Diesel/Petrol/CNG/LPG): ")
seller_type = input("Enter seller type (Individual/Dealer/Trustmark Dealer): ")
transmission = input("Enter transmission (Manual/Automatic): ")
owner = input("Enter owner type (First Owner/Second Owner/Third Owner/Fourth & Above Owner/Test Drive Car): ")
mileage = float(input("Enter mileage (km/l): "))
engine = float(input("Enter engine size (CC): "))
max_power = float(input("Enter max power (bhp): "))
seats = float(input("Enter number of seats: "))
brand = input("Enter car brand: ")

car_age = 2026 - year

# Create input data
new_car = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats],
    "car_age": [car_age],
    "brand": [brand]
})

# Predict price
predicted_price = pipeline.predict(new_car)

print("\n==============================")
print("PREDICTED CAR SELLING PRICE")
print("==============================")
print("₹", round(predicted_price[0], 2))