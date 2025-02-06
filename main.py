# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from prophet import Prophet
#
# # Function to load data
# @st.cache_data
# def load_data():
#     file_path = "data.csv"
#     return pd.read_csv(file_path, parse_dates=["Date"])
#
# df = load_data()
#
# # Function to extract base supplier names dynamically
# def extract_base_supplier(supplier_name):
#     base_names = [
#         "protea by marriott", "orion safari", "anew hotel", "anew resort",
#         "city lodge hotel", "the capital", "garden court", "stay easy",
#         "town lodge", "sun city", "southern sun", "hilton", "marriott",
#         "holiday inn", "safari lodge", "palm swift", "be@home"
#     ]
#     supplier_name = str(supplier_name).lower()
#     for base in base_names:
#         if base in supplier_name:
#             return base.title()
#     return supplier_name.title()
#
# # Apply function to extract base supplier names
# df['Base_Supplier'] = df['Supplier'].apply(extract_base_supplier)
#
# # Streamlit UI
# st.title("📊 Supplier Time Series Forecasting")
#
# # Dynamic Supplier Selection
# unique_suppliers = df["Base_Supplier"].unique()
# selected_supplier = st.selectbox("Select a Supplier", unique_suppliers)
#
# # Forecast period slider
# months_to_predict = st.slider("Months to Predict", min_value=1, max_value=12, value=6)
#
# # Filter data for the selected supplier
# supplier_data = df[df["Base_Supplier"] == selected_supplier][["Date", "Zimasa VAT Inclusive"]]
#
# if supplier_data.empty:
#     st.warning("No data available for this supplier. Try another selection.")
# else:
#     supplier_data = supplier_data.rename(columns={"Date": "ds", "Zimasa VAT Inclusive": "y"})
#     model = Prophet()
#     model.fit(supplier_data)
#
#     future = model.make_future_dataframe(periods=months_to_predict, freq='M')
#     forecast = model.predict(future)
#
#     # Get the start and end of forecast period
#     forecast_start_date = supplier_data["ds"].max()  # Last actual data point
#     forecast_end_date = forecast["ds"].max()  # End of selected forecast period
#
#     # Plot results with Plotly
#     fig = go.Figure()
#
#     # Actual data - Bold line
#     fig.add_trace(go.Scatter(
#         x=supplier_data["ds"], y=supplier_data["y"],
#         mode="lines+markers", name="Actual Data",
#         line=dict(color="blue", width=3)
#     ))
#
#     # Predicted data - Dotted line
#     fig.add_trace(go.Scatter(
#         x=forecast["ds"], y=forecast["yhat"],
#         mode="lines+markers", name="Forecast",
#         line=dict(color="red", width=2)
#     ))
#
#     # Highlight only the forecasted period with a shaded background
#     fig.add_vrect(
#         x0=forecast_start_date, x1=forecast_end_date,
#         fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0
#     )
#
#     # Format Y-axis to whole numbers and limit range to actual data only
#     fig.update_layout(
#         title=f"Time Series Forecast for {selected_supplier}",
#         xaxis_title="Date",
#         yaxis_title="Zimasa VAT Inclusive",
#         hovermode="x",
#         yaxis=dict(
#             tickformat="d",  # Ensures whole numbers
#             # range=[min_actual, max_actual]  # Limits Y-axis to actual data range
#         )
#     )
#
#     st.plotly_chart(fig)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Load data
@st.cache_data
def load_data():
    file_path = "data.csv"
    return pd.read_csv(file_path, parse_dates=["Date"])

df = load_data()

# Extract base supplier names dynamically
def extract_base_supplier(supplier_name):
    base_names = [
        "protea by marriott", "orion safari", "anew hotel", "anew resort",
        "city lodge hotel", "the capital", "garden court", "stay easy",
        "town lodge", "sun city", "southern sun", "hilton", "marriott",
        "holiday inn", "safari lodge", "palm swift", "be@home"
    ]
    supplier_name = str(supplier_name).lower()
    for base in base_names:
        if base in supplier_name:
            return base.title()
    return supplier_name.title()

# Apply function to extract base supplier names
df['Base_Supplier'] = df['Supplier'].apply(extract_base_supplier)

# Streamlit UI
st.title("📊 Supplier Time Series Forecasting")

# Supplier Selection
unique_suppliers = df["Base_Supplier"].unique()
selected_supplier = st.selectbox("Select a Supplier", unique_suppliers)

# Forecast period slider
months_to_predict = st.slider("Months to Predict", min_value=1, max_value=12, value=6)

# Filter data for selected supplier
supplier_data = df[df["Base_Supplier"] == selected_supplier][["Date", "Zimasa VAT Inclusive"]]

if supplier_data.empty:
    st.warning("No data available for this supplier. Try another selection.")
else:
    # Rename columns for Prophet
    supplier_data = supplier_data.rename(columns={"Date": "ds", "Zimasa VAT Inclusive": "y"})

    # Convert 'y' to numeric and drop missing values
    supplier_data["y"] = pd.to_numeric(supplier_data["y"], errors="coerce")
    supplier_data.dropna(subset=["y"], inplace=True)  # Remove rows with NaN 'y'

    # Check if data is valid
    if supplier_data.empty:
        st.error("Error: No valid numeric data found in 'y'. Please check dataset.")
    else:
        # Set an upper bound for logistic growth
        cap_value = supplier_data["y"].max() * 1.5  # 50% higher than max value
        supplier_data["cap"] = cap_value

        # Prophet Model with logistic growth
        model = Prophet(growth="logistic", weekly_seasonality=True, yearly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Train the model
        model.fit(supplier_data)

        # Create future dataframe
        future = model.make_future_dataframe(periods=months_to_predict, freq='M')

        # Set cap for future data
        future["cap"] = cap_value  # Ensure future data has the same capacity limit

        # Predict future values
        forecast = model.predict(future)

        # Plot results with Plotly
        fig = go.Figure()

        # Actual data - Bold line
        fig.add_trace(go.Scatter(
            x=supplier_data["ds"], y=supplier_data["y"],
            mode="lines", name="Actual Data",
            line=dict(color="deepskyblue", width=3)
        ))

        # Predicted data - Dotted line
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines", name="Forecast",
            line=dict(color="red", width=2, dash="dot")
        ))

        # Highlight only the forecasted period
        forecast_start_date = supplier_data["ds"].max()
        forecast_end_date = forecast["ds"].max()
        fig.add_vrect(
            x0=forecast_start_date, x1=forecast_end_date,
            fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0
        )

        # Format Y-axis to whole numbers
        fig.update_layout(
            title=f"Time Series Forecast for {selected_supplier}",
            xaxis_title="Date",
            yaxis_title="Zimasa VAT Inclusive",
            hovermode="x",
            template="plotly_white",
            yaxis=dict(
                tickformat="d"  # Ensures whole numbers
            )
        )

        st.plotly_chart(fig)
