# ==============================================================================
# WEEK 2: ADVANCED DATA VISUALIZATION AND STORYTELLING WITH PYTHON
# Complete End-to-End Pipeline for Google Colab
# ==============================================================================

# Step 0: Ensure required packages are installed
!pip install pandas numpy matplotlib seaborn plotly openpyxl -q

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Set Global Plot Configurations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

print("✅ Setup complete. Initializing synthetic e-commerce data pipeline...")

# ==============================================================================
# STEP 1: DATA SYNTHESIS & INGESTION (Simulating Real E-Commerce Records)
# ==============================================================================
np.random.seed(42)
n_records = 15000

categories = [
    "Electronics",
    "Home & Kitchen",
    "Apparel",
    "Beauty & Health",
    "Books",
    "Sports",
]
countries = [
    "United States",
    "United Kingdom",
    "Germany",
    "France",
    "Canada",
    "Australia",
    "Japan",
]
regions = {
    "United States": "North America",
    "Canada": "North America",
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Australia": "APAC",
    "Japan": "APAC",
}

# Generate random components
cust_ids = np.random.choice(
    [f"CUST_{i:04d}" for i in range(1, 1200)], size=n_records
)
# Inject ~5% guest transactions (missing customer ID)
cust_ids = [cid if np.random.rand() > 0.05 else np.nan for cid in cust_ids]

dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="min")
invoice_dates = np.random.choice(dates, size=n_records)
chosen_cats = np.random.choice(
    categories, size=n_records, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
)
chosen_countries = np.random.choice(
    countries, size=n_records, p=[0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05]
)

quantities = np.random.choice(
    [1, 2, 3, 4, 5, 8, 10, 20, 50, -1, -5],
    size=n_records,
    p=[0.40, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.015, 0.005, 0.005, 0.005],
)
base_prices = {
    "Electronics": 180.0,
    "Home & Kitchen": 75.0,
    "Apparel": 45.0,
    "Beauty & Health": 30.0,
    "Books": 18.0,
    "Sports": 55.0,
}
unit_prices = [
    np.round(
        np.random.normal(loc=base_prices[cat], scale=base_prices[cat] * 0.25), 2
    )
    for cat in chosen_cats
]

raw_df = pd.DataFrame(
    {
        "InvoiceNo": [f"INV_{100000 + i}" for i in range(n_records)],
        "CustomerID": cust_ids,
        "InvoiceDate": invoice_dates,
        "Category": chosen_cats,
        "Quantity": quantities,
        "UnitPrice": unit_prices,
        "Country": chosen_countries,
    }
)
raw_df["Region"] = raw_df["Country"].map(regions)

# Introduce 150 exact duplicate rows to replicate raw data challenges
duplicates = raw_df.sample(150, random_state=42)
raw_df = pd.concat([raw_df, duplicates], ignore_index=True)

print(f"Dataset generated: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")

# ==============================================================================
# STEP 2: DATA CLEANING & REFINEMENT
# ==============================================================================
df_clean = raw_df.copy()

# 1. Deduplication
df_clean = df_clean.drop_duplicates()

# 2. Type Casting & Datetime Transformation
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

# 3. Handle Missing IDs
df_clean["CustomerID"] = df_clean["CustomerID"].fillna("Guest")

# 4. Outlier & Negative Filter (Exclude returns & zero/negative pricing)
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# 5. Feature Engineering: Gross Transaction Total
df_clean["TotalAmount"] = np.round(
    df_clean["Quantity"] * df_clean["UnitPrice"], 2
)

print(f"Data cleaned: {len(df_clean)} verified transactional records ready.\n")

# ==============================================================================
# VISUALIZATION 1: CUMULATIVE REVENUE LORENZ CURVE (PARETO 80/20 ANALYSIS)
# ==============================================================================
print("Generating Visualization 1: Cumulative Revenue Lorenz Curve...")

registered_custs = df_clean[df_clean["CustomerID"] != "Guest"]
customer_spend = (
    registered_custs.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
customer_spend["CumulativeSpend"] = customer_spend["TotalAmount"].cumsum()
customer_spend["CumulativePct"] = (
    100 * customer_spend["CumulativeSpend"] / customer_spend["TotalAmount"].sum()
)
customer_spend["CustomerPct"] = (
    100 * (customer_spend.index + 1) / len(customer_spend)
)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    customer_spend["CustomerPct"],
    customer_spend["CumulativePct"],
    color="#0d6efd",
    linewidth=2.5,
    label="Observed Customer Spend",
)
ax.plot(
    [0, 100],
    [0, 100],
    color="#dc3545",
    linestyle="--",
    linewidth=1.5,
    label="Equal Spend Baseline (Hypothetical)",
)

# 20% Customer marker
idx_20 = int(len(customer_spend) * 0.20)
rev_20 = customer_spend.iloc[idx_20]["CumulativePct"]
ax.scatter([20], [rev_20], color="#dc3545", s=90, zorder=5)
ax.annotate(
    f"Top 20% Customers = {rev_20:.1f}% Total Revenue",
    xy=(20, rev_20),
    xytext=(35, rev_20 - 15),
    arrowprops=dict(facecolor="#212529", shrink=0.08, width=1, headwidth=6),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ced4da", lw=1),
)

ax.set_title(
    "Revenue Concentration: Cumulative Customer Spend (Lorenz Curve)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Percentage of Customer Base (%)", fontsize=11)
ax.set_ylabel("Share of Cumulative Revenue (%)", fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("chart_1_lorenz_curve.png", dpi=300)
plt.show()

# ==============================================================================
# VISUALIZATION 2: WEEKLY PURCHASING HEATMAP (OPERATIONAL DEMAND TIMING)
# ==============================================================================
print("Generating Visualization 2: Weekly Revenue Density Heatmap...")

df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
days_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

heatmap_data = (
    df_clean.groupby(["DayOfWeek", "Hour"])["TotalAmount"]
    .sum()
    .unstack(fill_value=0)
    .reindex(days_order)
)

fig, ax = plt.subplots(figsize=(12, 5.5))
sns.heatmap(
    heatmap_data,
    cmap="Blues",
    cbar_kws={"label": "Gross Sales ($)"},
    linewidths=0.25,
    linecolor="#e9ecef",
    ax=ax,
)

ax.set_title(
    "Weekly Revenue Density: Hour of Day vs. Day of Week",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Hour of Day (24h Clock)", fontsize=11)
ax.set_ylabel("Day of Week", fontsize=11)
plt.tight_layout()
plt.savefig("chart_2_weekly_heatmap.png", dpi=300)
plt.show()

# ==============================================================================
# VISUALIZATION 3: MONTHLY CUSTOMER RETENTION COHORT MATRIX
# ==============================================================================
print("Generating Visualization 3: Customer Cohort Retention Heatmap...")

registered_only = df_clean[df_clean["CustomerID"] != "Guest"].copy()
registered_only["OrderMonth"] = registered_only["InvoiceDate"].dt.to_period("M")
registered_only["CohortMonth"] = (
    registered_only.groupby("CustomerID")["InvoiceDate"]
    .transform("min")
    .dt.to_period("M")
)

cohort_df = (
    registered_only.groupby(["CohortMonth", "OrderMonth"])["CustomerID"]
    .nunique()
    .reset_index()
)
cohort_df["CohortIndex"] = (
    (cohort_df["OrderMonth"].dt.year - cohort_df["CohortMonth"].dt.year) * 12
    + (cohort_df["OrderMonth"].dt.month - cohort_df["CohortMonth"].dt.month)
)

cohort_matrix = cohort_df.pivot(
    index="CohortMonth", columns="CohortIndex", values="CustomerID"
)
cohort_sizes = cohort_matrix.iloc[:, 0]
retention_matrix = (
    cohort_matrix.divide(cohort_sizes, axis=0) * 100
).round(1)

fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(
    retention_matrix,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "Retention Rate (%)"},
    ax=ax,
)

ax.set_title(
    "Customer Cohort Retention Rate (%) Across Lifecycle Months",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Months Since First Acquisition", fontsize=11)
ax.set_ylabel("Acquisition Cohort (Year-Month)", fontsize=11)
plt.tight_layout()
plt.savefig("chart_3_cohort_retention.png", dpi=300)
plt.show()

# ==============================================================================
# VISUALIZATION 4: DUAL-AXIS CATEGORY PERFORMANCE (REVENUE VS. AOV)
# ==============================================================================
print("Generating Visualization 4: Category Margin vs. Revenue Breakdown...")

cat_perf = (
    df_clean.groupby("Category")
    .agg(
        TotalRevenue=("TotalAmount", "sum"),
        AvgBasketValue=("TotalAmount", "mean"),
    )
    .reset_index()
    .sort_values(by="TotalRevenue", ascending=False)
)

fig, ax1 = plt.subplots(figsize=(10, 5.5))

color_bar = "#34495e"
color_line = "#e67e22"

# Primary Axis: Total Revenue ($K)
ax1.bar(
    cat_perf["Category"],
    cat_perf["TotalRevenue"] / 1000,
    color=color_bar,
    alpha=0.85,
    width=0.45,
    label="Total Revenue ($K)",
)
ax1.set_ylabel("Total Revenue ($ in Thousands)", color=color_bar, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_bar)
ax1.set_xticklabels(cat_perf["Category"], rotation=20, ha="right", fontsize=10)

# Secondary Axis: Average Order Value
ax2 = ax1.twinx()
ax2.plot(
    cat_perf["Category"],
    cat_perf["AvgBasketValue"],
    color=color_line,
    marker="o",
    linewidth=2.5,
    markersize=7,
    label="Avg Order Value ($)",
)
ax2.set_ylabel("Average Order Value ($)", color=color_line, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color_line)
ax2.grid(False)

plt.title(
    "Product Category Performance: Total Revenue vs. Average Order Basket Value",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
fig.tight_layout()
plt.savefig("chart_4_category_performance.png", dpi=300)
plt.show()

# ==============================================================================
# VISUALIZATION 5: INTERACTIVE GLOBAL REGIONAL PERFORMANCE (PLOTLY BUBBLE MAP)
# ==============================================================================
print("Generating Visualization 5: Interactive Regional Performance Map...")

regional_summary = (
    df_clean.groupby(["Country", "Region"])
    .agg(
        TotalSales=("TotalAmount", "sum"),
        OrderCount=("InvoiceNo", "count"),
        AvgOrderSpend=("TotalAmount", "mean"),
    )
    .reset_index()
)

fig = px.scatter_geo(
    regional_summary,
    locations="Country",
    locationmode="country names",
    color="Region",
    size="TotalSales",
    hover_name="Country",
    hover_data={
        "TotalSales": ":$,.2f",
        "OrderCount": ":,",
        "AvgOrderSpend": ":$,.2f",
        "Country": False,
    },
    projection="natural earth",
    title="<b>Global Revenue Footprint & Market Depth by Country</b>",
    template="plotly_white",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02),
    height=550,
)

# Display interactive map in notebook
fig.show()

# Export interactive plot as standalone HTML
fig.write_html("chart_5_global_sales_map.html")
print("Saved interactive map as 'chart_5_global_sales_map.html'.")

# ==============================================================================
# CONFIRMATION OF SAVED OUTPUTS
# ==============================================================================
print("\n" + "=" * 60)
print("🎉 ALL 5 CHARTS PRODUCED AND EXPORTED SUCCESSFULLY!")
print("Saved Visual Assets in Colab Files Panel:")
for file in [
    "chart_1_lorenz_curve.png",
    "chart_2_weekly_heatmap.png",
    "chart_3_cohort_retention.png",
    "chart_4_category_performance.png",
    "chart_5_global_sales_map.html",
]:
    if os.path.exists(file):
        print(f"  • {file}")
print("=" * 60)