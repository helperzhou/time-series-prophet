import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_highcharts as sh
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet


st.set_page_config(
    page_title='Quantilytix Data Exploration Platform'
    , page_icon='💻'
    , layout='wide'
)


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
st.title("📊 :red[Quantilytix] Data Exploration Platform")

options = option_menu(
    menu_title=None
    , options=['Visualisations', 'Forecasting']
    , icons=['clipboard', 'line']
    , menu_icon='cast'
    , default_index=0
    , orientation='horizontal'
)

if options == 'Visualisations':

    # --- Chart Selection ---
    st.write("### 📊 Select Chart to Display")
    chart_option = st.selectbox(
        "Choose a Chart Type",
        [
            "VAT Overtime", "Supplier Spending", "VAT Contribution",
            "Trend Analysis", "Top Suppliers", "Pareto Chart"
        ]
    )

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric columns
    for col in ["VAT Exclusive", "VAT", "Zimasa VAT Inclusive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaN values for relevant charts
    df = df.dropna(subset=["Zimasa VAT Inclusive"])

    # Remove "Total" entries from Supplier column
    df = df[df["Supplier"].str.lower() != "total"]

    # === TIME SERIES ANALYSIS (Line Chart) ===
    if chart_option == "VAT Overtime":
        time_series_data = df.groupby("Date")["Zimasa VAT Inclusive"].sum().reset_index()
        time_chart_config = {
            "chart": {"type": "line"},
            "title": {"text": "Zimasa VAT Inclusive Over Time"},
            "xAxis": {"categories": time_series_data["Date"].astype(str).tolist()},
            "yAxis": {"title": {"text": "Amount (ZAR)"}},
            "series": [{"name": "Zimasa VAT Inclusive", "data": time_series_data["Zimasa VAT Inclusive"].tolist()}],
        }
        sh.streamlit_highcharts(time_chart_config)

    # === SUPPLIER SPENDING (Bar Chart) ===
    elif chart_option == "Supplier Spending":
        supplier_spending = df.groupby("Supplier")["Zimasa VAT Inclusive"].sum().reset_index()
        supplier_spending = supplier_spending.sort_values("Zimasa VAT Inclusive", ascending=False).head(10)

        supplier_chart_config = {
            "chart": {"type": "column"},
            "title": {"text": "Top 10 Suppliers by Spending"},
            "xAxis": {"categories": supplier_spending["Supplier"].tolist()},
            "yAxis": {"title": {"text": "Amount (ZAR)"}},
            "series": [{"name": "Spending", "data": supplier_spending["Zimasa VAT Inclusive"].tolist()}],
        }
        sh.streamlit_highcharts(supplier_chart_config)

    # === VAT CONTRIBUTION (Pie Chart) ===
    elif chart_option == "VAT Contribution":
        vat_data = df.groupby("Supplier")["VAT"].sum().reset_index().sort_values("VAT", ascending=False).head(10)
        vat_data["Percentage"] = (vat_data["VAT"] / vat_data["VAT"].sum()) * 100

        vat_chart_config = {
            "chart": {"type": "pie"},
            "title": {"text": "Top 10 VAT Contributions by Supplier"},
            "series": [
                {
                    "name": "VAT Contribution (%)",
                    "data": [{"name": row["Supplier"], "y": row["Percentage"]} for _, row in vat_data.iterrows()],
                }
            ],
        }
        sh.streamlit_highcharts(vat_chart_config)

    # === TREND ANALYSIS (100% Stacked Column Chart) ===
    elif chart_option == "Trend Analysis":
        trend_data = df.groupby("Date")[["VAT Exclusive", "VAT", "Zimasa VAT Inclusive"]].sum().reset_index()

        trend_chart_config = {
            "chart": {"type": "column"},
            "title": {"text": "VAT Breakdown Over Time"},
            "xAxis": {"categories": trend_data["Date"].astype(str).tolist()},
            "yAxis": {"title": {"text": "Percentage"}},
            "plotOptions": {"column": {"stacking": "percent"}},
            "series": [
                {"name": "VAT Exclusive", "data": trend_data["VAT Exclusive"].tolist()},
                {"name": "VAT", "data": trend_data["VAT"].tolist()},
                {"name": "VAT Inclusive", "data": trend_data["Zimasa VAT Inclusive"].tolist()},
            ],
        }
        sh.streamlit_highcharts(trend_chart_config)

    # === TOP SUPPLIERS (Bar Chart) ===
    elif chart_option == "Top Suppliers":
        supplier_spending = df.groupby("Supplier")["Zimasa VAT Inclusive"].sum().reset_index()
        supplier_spending = supplier_spending.sort_values("Zimasa VAT Inclusive", ascending=False).head(10)

        bar_chart_config = {
            "chart": {"type": "bar"},
            "title": {"text": "Top 10 Suppliers by Spending"},
            "xAxis": {"categories": supplier_spending["Supplier"].tolist()},
            "yAxis": {"title": {"text": "Spending (ZAR)"}},
            "series": [{"name": "Spending", "data": supplier_spending["Zimasa VAT Inclusive"].tolist()}],
        }
        sh.streamlit_highcharts(bar_chart_config)

    # === PARETO CHART (Bar + Line Combination) ===
    elif chart_option == "Pareto Chart":
        supplier_spending = df.groupby("Supplier")["Zimasa VAT Inclusive"].sum().reset_index()
        supplier_spending = supplier_spending.sort_values("Zimasa VAT Inclusive", ascending=False).head(15)

        supplier_spending["Cumulative %"] = supplier_spending["Zimasa VAT Inclusive"].cumsum() / supplier_spending[
            "Zimasa VAT Inclusive"].sum() * 100

        pareto_chart_config = {
            "title": {"text": "Pareto Chart of Top 10 Suppliers"},
            "xAxis": {"categories": supplier_spending["Supplier"].tolist()},
            "yAxis": [
                {"title": {"text": "Spending (ZAR)"}, "opposite": False},
                {"title": {"text": "Cumulative %"}, "opposite": True, "min": 0, "max": 100},
            ],
            "series": [
                {"type": "column", "name": "Spending", "data": supplier_spending["Zimasa VAT Inclusive"].tolist()},
                {"type": "line", "name": "Cumulative %", "data": supplier_spending["Cumulative %"].tolist(),
                 "yAxis": 1},
            ],
        }
        sh.streamlit_highcharts(pareto_chart_config)
elif options == 'Forecasting':
    # Filter suppliers that appear 2 or more times
    supplier_counts = df["Supplier"].value_counts()
    valid_suppliers = supplier_counts[supplier_counts >= 2].index

    # Extract only these valid suppliers from the dataset
    filtered_df = df[df["Supplier"].isin(valid_suppliers)]

    # Supplier Selection
    unique_suppliers = filtered_df["Supplier"].unique()
    selected_supplier = st.selectbox("Select a Supplier", unique_suppliers)

    # Forecast period slider
    months_to_predict = st.slider("Months to Predict", min_value=1, max_value=12, value=6)

    # Filter data for selected supplier
    supplier_data = filtered_df[filtered_df["Supplier"] == selected_supplier][["Date", "Zimasa VAT Inclusive"]]

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
