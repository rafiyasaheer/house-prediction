import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\internship\Linear-Regression-on-Housing-Data\train.csv")


features = ["area", "bedrooms", "bathrooms"]
X = data[features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏡 House Price Predictor")
st.write("This app predicts house prices using a Linear Regression model.")

# Display dataset information
if st.checkbox("Show Dataset"):
    st.write("### Dataset")
    st.dataframe(data.head())

# Model Evaluation Metrics
st.write("### Model Performance Metrics")
st.write(f"**Mean Squared Error: **{mse:.2f}**")
st.write(f"**R-Squared Value: **{r2:.2f}**")

# Plot Actual vs Predicted Prices
st.write("### Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual Prices vs Predicted Prices")
st.pyplot(fig)

# Prediction Interface
st.write("### Predict House Price")
area = st.number_input("Enter Ground Living Area (in sq ft)", min_value=0, value=2000)
bathrooms = st.number_input("Enter Number of Full Bathrooms", min_value=0, value=2)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=0, value=3)

if st.button("Predict House Price"):
    new_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })
    predicted_price = model.predict(new_data)
    st.success(f"Predicted Price: ${predicted_price[0]:,.2f}**")
st.write("### Dollar to Rupee Converter")
dollar_amount = st.number_input("Enter amount in USD", min_value=0.0, value=1.0)
exchange_rate = 82.0  # Approximate exchange rate
rupee_amount = dollar_amount * exchange_rate
if st.button("Convert to Rupees"):
    st.success(f"${dollar_amount} is approximately: ₹{rupee_amount:,.2f}")

    

