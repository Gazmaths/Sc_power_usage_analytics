# energy_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -----------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------
st.set_page_config(
    page_title="Southern Company Energy Dashboard",
    page_icon="Southern_company_logo.png",
    layout="wide"
)

import base64

# Load and convert image to base64
def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Base64-encoded image string
img_base64 = image_to_base64("Southern_company_logo.png")

# HTML with inline image and styled header
southern_blue = "#002554"
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/png;base64,{img_base64}" alt="Logo" width="50" style="margin-bottom: 5px;">
        <h1 style="color:{southern_blue}; margin: 0;">Southern Company Energy Usage Dashboard for 2024</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    usage_df = pd.read_excel("Hourly_usage.xlsx")
    weather_df = pd.read_excel("Weather.xlsx")

    usage_df['date'] = pd.to_datetime(usage_df['date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_df.rename(columns={"Date": "date", "Hour": "hour", "DRY_BULB_TEMP": "temperature"}, inplace=True)

    merged = pd.merge(usage_df, weather_df, on=["date", "hour"])
    return usage_df, weather_df, merged

usage_df, weather_df, merged_df = load_data()

# -----------------------------
# CUSTOMER METADATA
# -----------------------------
account_info = pd.DataFrame({
    "Account": [
        "id_865664259", "id_502927869", "id_556438708", "id_797210973", "id_159394858", "id_398371246",
        "id_622078427", "id_700052256", "id_459726488", "id_247562058", "id_423657958", "id_339238458",
        "id_867297720", "id_627637245", "id_400133509", "id_424331807", "id_680217490", "id_126971710",
        "id_935947466", "id_876648295", "id_700971314", "id_816335515", "id_833953115", "id_301719675",
        "id_236845199", "id_496557256"
    ],
    "Type": [
        "Commercial", "Industrial", "Commercial", "Commercial", "Commercial", "Residential",
        "Residential", "Industrial", "Industrial", "Residential", "Commercial", "Commercial",
        "Residential", "Industrial", "Industrial", "Residential", "Commercial", "Residential",
        "Residential", "Commercial", "Residential", "Commercial", "Industrial", "Commercial",
        "Residential", "Residential"
    ],
    "Rate": [
        "TOU-FD", "RTHPLL", "TOU-FD", "TOU-FD", "TOU-FD", "R",
        "R", "RTHPLL", "TOU-SC", "R", "TOU-FD", "TOU-FD",
        "R", "RTHPLL", "RTDPLM", "R", "TOU-FD", "R",
        "R", "TOU-FD", "R", "TOU-FD", "RTHTOURN", "TOU-FD",
        "R", "R"
    ]
})

# Melt and merge
usage_long = usage_df.melt(id_vars=["date", "hour"], var_name="Account", value_name="Usage")
usage_long = usage_long.merge(account_info, on="Account")

merged_long = merged_df.melt(id_vars=["date", "hour", "temperature"], var_name="Account", value_name="Usage")
merged_long = merged_long.merge(account_info, on="Account")

# -----------------------------
# SELECTOR AT TOP
# -----------------------------
selected_type = st.selectbox("Select Customer Type to View", ["All", "Residential", "Commercial", "Industrial"])

if selected_type != "All":
    usage_filtered = usage_long[usage_long['Type'] == selected_type]
    merged_filtered = merged_long[merged_long['Type'] == selected_type]
else:
    usage_filtered = usage_long
    merged_filtered = merged_long

# -----------------------------
# KPIs
# -----------------------------
total_usage = usage_filtered['Usage'].sum()
avg_daily = usage_filtered.groupby('date')['Usage'].sum().mean()
avg_monthly = usage_filtered.assign(month=usage_filtered['date'].dt.to_period("M")).groupby('month')['Usage'].sum().mean()
num_customers = usage_filtered['Account'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Energy Used (MWh)", f"{total_usage/1000:,.2f}")
col2.metric("Avg Daily Usage (MWh)", f"{avg_daily/1000:.2f}")
col3.metric("Avg Monthly Usage (MWh)", f"{avg_monthly/1000:.2f}")
col4.metric("Total Customers", f"{num_customers}")

# -----------------------------
# USAGE BY TYPE AND RATE
# -----------------------------
by_type = usage_filtered.groupby('Type')['Usage'].sum().reset_index()
fig1 = px.pie(by_type, names='Type', values='Usage', title='Usage by Customer Type')
fig1.update_traces(marker=dict(colors=[southern_blue, '#007BFF', '#6699CC']), hole=0.4)

by_rate = usage_filtered.groupby('Rate')['Usage'].sum().reset_index()
fig2 = px.bar(by_rate, x='Rate', y='Usage', title='Usage by Rate Category', color='Rate')
fig2.update_layout(showlegend=False)

col5, col6 = st.columns(2)
col5.plotly_chart(fig1, use_container_width=True)
col6.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TOP 5 ENERGY-CONSUMING ACCOUNTS
# -----------------------------
top_accounts = usage_filtered.groupby('Account')['Usage'].sum().nlargest(5).reset_index()
fig3 = px.bar(top_accounts, x='Usage', y='Account', orientation='h', title='Top 5 Energy-Consuming Accounts', color='Account')
fig3.update_layout(showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# HOURLY USAGE PATTERN
# -----------------------------
hourly_avg = usage_filtered.groupby('hour')['Usage'].mean().reset_index()
fig4 = px.line(hourly_avg, x='hour', y='Usage', title='Average Hourly Energy Usage', markers=True)
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# SEASONAL USAGE
# -----------------------------
usage_filtered['month'] = usage_filtered['date'].dt.strftime('%B')
seasonal_avg = usage_filtered.groupby('month')['Usage'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]).reset_index()
fig5 = px.bar(seasonal_avg, x='month', y='Usage', title='Average Monthly Usage')
st.plotly_chart(fig5, use_container_width=True)



# Map month to season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Add season column
usage_filtered['season'] = usage_filtered['date'].dt.month.apply(get_season)
seasonal_usage = usage_filtered.groupby('season')['Usage'].sum().reindex(['Spring', 'Summer', 'Fall', 'Winter']) / 1e6

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(seasonal_usage.index, seasonal_usage.values, color='red')

# Title and labels
ax.set_title('Total Electricity Usage by Season', fontsize=14, weight='bold')
ax.set_ylabel('Total Usage (in Millions kWh)', fontsize=12)
ax.set_xlabel('Season', fontsize=12)
ax.set_ylim(0, seasonal_usage.max() * 1.2)

sns.despine()
st.pyplot(fig)


# REGRESSION ANALYSIS USING MATPLOTLIB - NO GROUPING, JUST ALIGN ROWS

total_usage_hourly = merged_df.drop(columns='temperature').melt(id_vars=['date', 'hour'], var_name='Account', value_name='Usage')
total_usage_hourly = total_usage_hourly.groupby(['date', 'hour'], as_index=False)['Usage'].sum()

temperature_hourly = weather_df[['date', 'hour', 'temperature']]

reg_data = pd.merge(total_usage_hourly, temperature_hourly, on=['date', 'hour'])

# Fit linear model
X = reg_data['temperature'].values.reshape(-1, 1)
y = reg_data['Usage'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Step 5: Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='temperature', y='Usage', data=reg_data, color='red', s=10, ax=ax)
sns.lineplot(x=reg_data['temperature'], y=y_pred, color='blue', ax=ax)

ax.set_title("Regression of Total Electricity Usage vs Temperature\n$R^2$ = {:.4f}".format(r2), fontsize=14, weight='bold')
ax.set_xlabel("Dry Bulb Temperature (°F)", fontsize=12)
ax.set_ylabel("Total Electricity Usage (kWh)", fontsize=12)
ax.grid(False)
sns.despine()

st.pyplot(fig)

