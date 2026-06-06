import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def clean_data(df):
    df = df.drop_duplicates()

    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    numeric_columns = ["Sales", "Cost", "Profit", "Quantity", "Discount"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Sales" in df.columns and "Profit" in df.columns:
        df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100
        df["Profit_Margin"] = df["Profit_Margin"].replace(
            [float("inf"), -float("inf")], 0
        ).fillna(0)

    return df


def generate_insights(df):
    insights = []

    if "Region" in df.columns and "Sales" in df.columns:
        region_sales = df.groupby("Region")["Sales"].sum()
        top_region = region_sales.idxmax()
        top_region_sales = region_sales.max()

        insights.append(
            f"{top_region} is the highest revenue region with sales of ${top_region_sales:,.0f}."
        )

    if "Product" in df.columns and "Profit" in df.columns:
        product_profit = df.groupby("Product")["Profit"].sum()
        top_product = product_profit.idxmax()
        top_product_profit = product_profit.max()

        insights.append(
            f"{top_product} is the most profitable product with profit of ${top_product_profit:,.0f}."
        )

    if "Category" in df.columns and "Profit_Margin" in df.columns:
        category_margin = df.groupby("Category")["Profit_Margin"].mean()
        low_margin_category = category_margin.idxmin()
        low_margin_value = category_margin.min()

        insights.append(
            f"{low_margin_category} has the lowest average profit margin at {low_margin_value:.2f}%."
        )

    if "Month" in df.columns and "Sales" in df.columns:
        monthly_sales = df.groupby("Month")["Sales"].sum()

        best_month = monthly_sales.idxmax()
        best_month_sales = monthly_sales.max()

        insights.append(
            f"{best_month} was the best sales month with total sales of ${best_month_sales:,.0f}."
        )

    return insights


def forecast_sales(df):
    if "Month" not in df.columns or "Sales" not in df.columns:
        return None

    monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()
    monthly_sales = monthly_sales.sort_values("Month")

    if len(monthly_sales) < 2:
        return None

    monthly_sales["Month_Number"] = range(1, len(monthly_sales) + 1)

    X = monthly_sales[["Month_Number"]]
    y = monthly_sales["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[len(monthly_sales) + 1]])
    prediction = model.predict(next_month)[0]

    return prediction


st.set_page_config(page_title="AutoInsight Dashboard", layout="wide")

st.title("AutoInsight Dashboard")
st.subheader("Upload a CSV file to generate business insights automatically")

uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Charts",
        "Insights",
        "Forecast"
    ])

    with tab1:
        st.write("Data Preview")
        st.dataframe(df.head())

        st.subheader("Executive Summary")

        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        total_orders = df["Order_ID"].nunique()
        total_customers = df["Customer_ID"].nunique()
        profit_margin = (total_profit / total_sales) * 100

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Sales", f"${total_sales:,.0f}")
        col2.metric("Total Profit", f"${total_profit:,.0f}")
        col3.metric("Orders", total_orders)
        col4.metric("Customers", total_customers)
        col5.metric("Profit Margin", f"{profit_margin:.2f}%")

    with tab2:
        st.subheader("Business Charts")

        monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()

        st.line_chart(
            monthly_sales,
            x="Month",
            y="Sales"
        )

        region_sales = df.groupby("Region")["Sales"].sum().reset_index()

        st.bar_chart(
            region_sales,
            x="Region",
            y="Sales"
        )

        category_profit = df.groupby("Category")["Profit"].sum().reset_index()

        st.bar_chart(
            category_profit,
            x="Category",
            y="Profit"
        )

    with tab3:
        st.subheader("Automated Business Insights")

        insights = generate_insights(df)

        for insight in insights:
            st.write(f"- {insight}")

    with tab4:
        st.subheader("Sales Forecast")

        predicted_sales = forecast_sales(df)

        if predicted_sales:
            st.metric("Forecasted Next Month Sales", f"${predicted_sales:,.0f}")
        else:
            st.info("Need at least 2 months of sales data for forecasting.")
    
else:
    st.info("Please upload a CSV file to begin.")

