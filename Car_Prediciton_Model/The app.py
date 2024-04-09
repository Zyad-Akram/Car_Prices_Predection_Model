##### Importing the libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

##### Load the dataset
dataset = pd.read_csv(r'd:\ITI\Project ITI\iti\Car_sale_ads.csv')

##### Drop rows with missing values
dataset.dropna(inplace=True)

##### Select relevant columns for prediction
selected_columns = ['Condition', 'Vehicle_brand', 'Vehicle_model', 'Production_year', 'Mileage_km', 'Power_HP', 'Displacement_cm3', 'Fuel_type', 'Transmission', 'Type', 'Doors_number', 'Colour']
x = dataset[selected_columns]
y = dataset['Price']

##### Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

##### Preprocessing pipeline
numeric_features = ['Production_year', 'Mileage_km', 'Power_HP', 'Displacement_cm3', 'Doors_number']
categorical_features = ['Condition', 'Vehicle_brand', 'Vehicle_model', 'Fuel_type', 'Transmission', 'Type', 'Colour']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

##### Create and fit the model
regressor = LinearRegression()
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor)])
model.fit(x_train, y_train)

# Predicting the Test set results
y_pred_test = model.predict(x_test)

joblib.dump(model, 'Ultimate Model.pkl')

# Load the trained model
loaded_model = joblib.load('Ultimate Model.pkl')

# Define the UI function
def predict_price():
    user_inputs = {}
    for column in selected_columns:
        user_input = entry_vars[column].get()
        user_inputs[column] = [user_input]

    user_df = pd.DataFrame(user_inputs)

    try:
        predicted_price = loaded_model.predict(user_df)[0]
        result_label.config(text=f"Predicted Price: {predicted_price:.2f}")
    except:
        messagebox.showerror("Error", "Prediction failed. Please check your inputs.")

# Load the dataset and selected columns
dataset = pd.read_csv(r'd:\ITI\Project ITI\iti\Car_sale_ads.csv')
selected_columns = ['Condition', 'Vehicle_brand', 'Vehicle_model', 'Production_year', 'Mileage_km', 'Power_HP', 'Displacement_cm3', 'Fuel_type', 'Transmission', 'Type', 'Doors_number', 'Colour']

# Create the UI window
root = tk.Tk()
root.title("Car Price Predictor")

# Create entry widgets and labels for each column
entry_vars = {}
for column in selected_columns:
    label = tk.Label(root, text=f"Enter {column}:")
    label.pack()
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var)
    entry.pack()
    entry_vars[column] = entry_var

# Create Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.pack()

# Create label to display the predicted price
result_label = tk.Label(root, text="")
result_label.pack()

# Start the UI event loop
root.mainloop()