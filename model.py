# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score

# %% [markdown]
# Data Loading

# %%
# Read the CSV file into pandas DataFrame
try:
    df = pd.read_csv("coin_BinanceCoin.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'coin_BinanceCoin.csv' was not found")
    print("Please make sure the file is in the same directory")
    exit()
df

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated rows")
print(df_duplicated)

# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)
print("Data Preprocessing Complete!")

# %% [markdown]
# Feature Engineering

# %%
# Select the features (independent variables) that will be used to make predictons
# We are using the "open", "high", and "low" prices to predict the closing prices
features = ["open","high","low"]

# Select the target (dependent variable) which is what we want to predict.
target = "close"

# Separate the DataFrame into features (X) and target variable (y)
X = df[features]
y = df[target]

# %% [markdown]
# Data Splitting

# %%
# Split the data into a training set and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Visualization before Training

# %%
# Create a figure for the plot with specific size
plt.figure(figsize=(10,6))

# Create a scatter plot to visualize the relationship between "open" and "close" prices
plt.scatter(df["open"],df["close"],color='red')
plt.title("Relationship between the Open and Close Prices (Before Training)")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %%
# Create a figure for the plot with specific size
plt.figure(figsize=(10,6))

# Create a scatter plot to visualize the relationship between "open" and "close" prices
plt.scatter(df["high"],df["close"],color='orange')
plt.title("Relationship between the High and Close Prices (Before Training)")
plt.xlabel("High Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %%
# Create a figure for the plot with specific size
plt.figure(figsize=(10,6))

# Create a scatter plot to visualize the relationship between "open" and "close" prices
plt.scatter(df["low"],df["close"],color='black',alpha=0.6)
plt.title("Relationship between the Low and Close Prices (Before Training)")
plt.xlabel("Low Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %% [markdown]
# Model Building and Training

# %%
# Initialize a dictionary to store different regression models
models = {
    "Linear Regression":LinearRegression(),
    "Ridge Regression":Ridge(alpha=0.1),
    "Lasso Regression":Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1,l1_ratio=0.5)
}

# Initialize a dictionary to store the evaluation results (MSE) for eac model
results ={}
best_model = None
lowest_mse = float("inf")

# Loop through each model in the dictionary to train, predict and evaluate it
for name,model in models.items():
    # Train the model using the training data (X_train,y_train)
    model.fit(X_train,y_train)

    # Use the trained model to make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE) to evaluate the model's performance
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    results[name] = mse

    # Check if the current model has a lowe MSE than the previous best model
    if mse < lowest_mse:
        lowest_mse = mse
        best_model = model

print("Model Comparison (Mean Squared Error)")
for name,mse in results.items():
    print(f"- {name}: MSE {mse:.4f}, R-Sqaured {r2:.4f}")  

# %% [markdown]
# Identify the best model

# %%
best_model_name = [name for name,model in models.items() if model == best_model][0]
print(f"The best model is: {best_model_name}")

# %% [markdown]
# Prediction

# %%
y_pred_best = best_model.predict(X_test)

# %% [markdown]
# Visualization after training

# %%
# Create a scatter plot to visulize how well the predicted values match the actual value
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_best,color="darkblue")
plt.title(f"Actual vs Predicted Close Prices ({best_model_name})")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")

# Plot a diagonal line (y=x) to represent a perfect prediction
# The closer the points are to this ine, the better the model's perfotmance
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],lw=2,color="red")
plt.grid(True)
plt.show()

# %% [markdown]
# User Input and Prediction

# %%
def predict_with_user_input(model):
    """
    This functon takes user input for open, high and low prices
    and uses the trained model to predict the close price
    """
    try:
        # Prompt the user to enter the required values.
        open_price = float(input("Enter the Open Price:"))
        high_price = float(input("Enter the High Price:"))
        low_price = float(input("Enter the Low Price:"))

        # Create a DataFrame from the user's input, which is the format the model expects
        user_data = pd.DataFrame([[open_price,high_price,low_price]],columns=features)

        # Make a Prediction using the trained model
        predicted_price = model.predict(user_data)

        # Print the predicted price formatted to two decimal places
        print(f"Based on you input, the predicted Close Price is: {predicted_price[0]:.2f}")
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function
predict_with_user_input(model)


