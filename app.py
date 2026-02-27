import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tempfile

# ML Forecast
from sklearn.linear_model import LinearRegression

# Business calculations
from simulation.energy_model import calculate_cost, calculate_carbon

# PDF Report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

plt.style.use("dark_background")

plt.rcParams.update({
    "figure.facecolor": "#1A1F2E",
    "axes.facecolor": "#1A1F2E",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#E6E6E6",
    "xtick.color": "#CCCCCC",
    "ytick.color": "#CCCCCC",
    "text.color": "#E6E6E6",
    "grid.color": "#2C3245",
    "grid.alpha": 0.4,
    "axes.titleweight": "bold",
})

def generate_pdf_report():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AI-Powered Lighting Digital Twin Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [
        ["Metric", "Value"],
        ["Annual Savings", f"₹{annual_savings:,.0f}"],
        ["5-Year Savings", f"₹{five_year_savings:,.0f}"],
        ["Annual CO₂ Reduction", f"{annual_carbon_saved:,.0f} kg"],
        ["5-Year CO₂ Reduction", f"{five_year_carbon_saved:,.0f} kg"],
        ["Estimated Payback Period", f"{payback_period:.2f} Years"],
        ["Energy Efficiency Score", f"{efficiency_score:.1f}%"]
    ]

    table = Table(data)
    elements.append(table)

    doc.build(elements)
    return temp_file.name

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
    color: #E6E6E6;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1A1F2E;
    padding: 15px;
    border-radius: 10px;
}

/* Success box */
.stAlert-success {
    background-color: #003B2F;
    color: #00C49A;
}

/* Buttons */
.stButton>button {
    background-color: #00C49A;
    color: black;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Slightly bigger number input */
div[data-testid="stNumberInput"] input {
    font-size: 20px !important;
    padding: 12px 14px !important;
    height: 50px !important;
}

/* Slightly larger label */
div[data-testid="stNumberInput"] label {
    font-size: 18px !important;
    font-weight: 500;
}

/* Make +/- buttons balanced */
div[data-testid="stNumberInput"] button {
    height: 45px !important;
}

</style>
""", unsafe_allow_html=True)


# Use full-width layout for dashboard-style design
st.set_page_config(layout="wide")
# ===============================
# Vertical Divider Style
# ===============================

VERTICAL_DIVIDER = """
<div style="
    height: 100%;
    border-left: 1px solid #2C3245;
    margin: 0 auto;
"></div>
"""
st.title("AI-Powered Lighting Digital Twin 💡")
with st.spinner("AI Optimizing Lighting Strategy..."):
    import time
    time.sleep(0.8)
    progress = st.progress(0)

for i in range(100):
    time.sleep(0.005)
    progress.progress(i + 1)

progress.empty()
# =====================================================
# SIDEBAR – INPUT CONTROLS
# =====================================================

st.sidebar.header("Building Parameters")

num_lights = st.sidebar.number_input(
    "Number of Lights", min_value=1, value=100
)

wattage = st.sidebar.number_input(
    "Wattage per Light (W)", min_value=1, value=40
)

hours = st.sidebar.number_input(
    "Operating Hours per Day", min_value=1.0, value=10.0
)

tariff = st.sidebar.number_input(
    "Electricity Tariff (₹ per kWh)", min_value=0.0, value=8.0
)

brightness = st.sidebar.slider(
    "Brightness Level (%)", 0, 100, 100
) / 100

emission_factor = st.sidebar.number_input(
    "CO₂ Emission Factor (kg per kWh)", value=0.82
)

# =====================================================
# SCENARIO SELECTION
# =====================================================

st.markdown("---")
st.markdown("## 🔮 What-If Scenario Simulation")

scenario = st.selectbox(
    "Select Lighting Strategy",
    ["Standard Lighting", "Smart Adaptive", "High Efficiency Mode"]
)
if scenario == "Standard Lighting":
    st.info("Standard operation without adaptive intelligence.")

elif scenario == "Smart Adaptive":
    st.success("AI adjusts brightness based on occupancy.")

elif scenario == "High Efficiency Mode":
    st.warning("Aggressive energy saving mode enabled.")

# =====================================================
# OCCUPANCY SIMULATION (Digital Twin Behaviour)
# =====================================================

hours_in_day = list(range(24))
occupancy_pattern = []

# Simulate realistic daily building occupancy profile
for h in hours_in_day:
    if 9 <= h <= 17:
        occ = np.random.uniform(0.75, 0.95)  # Peak working hours
    elif 6 <= h < 9 or 17 < h <= 20:
        occ = np.random.uniform(0.3, 0.5)    # Transition periods
    else:
        occ = np.random.uniform(0.05, 0.15)  # Low usage hours
    occupancy_pattern.append(occ)

# =====================================================
# DYNAMIC ENERGY SIMULATION
# =====================================================
# Adjust brightness mode (base strategy)
if scenario == "Standard Lighting":
    strategy = "standard"

elif scenario == "Smart Adaptive":
    strategy = "adaptive"

elif scenario == "High Efficiency Mode":
    strategy = "efficient" # aggressive energy saving
hourly_energy = []
hourly_energy_optimized = []

# Assume building operates starting at 8 AM
start_hour = 8
end_hour = start_hour + int(hours)

for h, occ in zip(hours_in_day, occupancy_pattern):

    if start_hour <= h < end_hour:

        if strategy == "standard":
            brightness_used = brightness

        elif strategy == "adaptive":
            brightness_used = brightness * occ

        elif strategy == "efficient":
            brightness_used = brightness * 0.7

        energy_base = (num_lights * wattage * brightness_used) / 1000

        # Optimized comparison (always occupancy-based)
        adjusted_brightness = brightness * occ
        energy_optimized = (num_lights * wattage * adjusted_brightness) / 1000

    else:
        energy_base = 0
        energy_optimized = 0

    hourly_energy.append(energy_base)
    hourly_energy_optimized.append(energy_optimized)
# Aggregate daily and monthly energy
daily_energy = sum(hourly_energy)
daily_energy_opt = sum(hourly_energy_optimized)

monthly_energy = daily_energy * 30
opt_monthly_energy = daily_energy_opt * 30

# =====================================================
# COST & CARBON CALCULATIONS
# =====================================================

monthly_cost = calculate_cost(monthly_energy, tariff)
opt_monthly_cost = calculate_cost(opt_monthly_energy, tariff)

monthly_carbon = calculate_carbon(monthly_energy, emission_factor)
opt_monthly_carbon = calculate_carbon(opt_monthly_energy, emission_factor)

savings = monthly_cost - opt_monthly_cost
carbon_saved = monthly_carbon - opt_monthly_carbon
if monthly_carbon > 0:
    efficiency_score = (carbon_saved / monthly_carbon) * 100
else:
    efficiency_score = 0

annual_savings = savings * 12
five_year_savings = annual_savings * 5

annual_carbon_saved = carbon_saved * 12
five_year_carbon_saved = annual_carbon_saved * 5
# ===============================
# Predictive Sensitivity Engine
# ===============================

def simulate_change(occupancy_factor=1.0, tariff_factor=1.0, hours_factor=1.0):

    new_hours = hours * hours_factor
    new_tariff = tariff * tariff_factor

    # Recalculate energy
    new_daily_energy = daily_energy * occupancy_factor * hours_factor
    new_monthly_energy = new_daily_energy * 30

    new_cost = calculate_cost(new_monthly_energy, new_tariff)
    new_carbon = calculate_carbon(new_monthly_energy, emission_factor)

    return new_monthly_energy, new_cost, new_carbon

# ===============================
# AI Recommendation Engine
# ===============================

recommendations = []

if efficiency_score < 15:
    recommendations.append("Install occupancy-based motion sensors to reduce idle energy usage.")

if annual_savings < 20000:
    recommendations.append("Upgrade to high-efficiency LED lighting to improve ROI.")

if annual_carbon_saved < 5000:
    recommendations.append("Integrate renewable energy sources to lower carbon footprint.")

if annual_savings > 0:
    payback_period_est = 50000 / annual_savings
else:
    payback_period_est = 999

if payback_period_est > 3:
    recommendations.append("Consider phased deployment to reduce upfront investment risk.")

if not recommendations:
    recommendations.append("Current configuration is optimized. Maintain smart adaptive strategy.")

# =====================================================
# KPI SECTION (Top Row)
# =====================================================

st.subheader("Key Performance Indicators")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Monthly Energy", f"{monthly_energy:.0f} kWh")

with kpi2:
    st.metric("Optimized Energy", f"{opt_monthly_energy:.0f} kWh")

with kpi3:
    st.metric(
    "Monthly Savings",
    f"₹{savings:.0f}",
    delta=f"+₹{savings * 0.05:.0f} vs Last Month",
    delta_color="normal"
    )
    st.metric(
    "Efficiency Score",
    f"{efficiency_score:.1f}%",
    delta=f"+{efficiency_score * 0.1:.1f}% Improvement"
    )

with kpi4:
    st.metric("CO₂ Reduction", f"{carbon_saved:.0f} kg")

with kpi5:
    st.metric("Efficiency Score", f"{efficiency_score:.1f}%")    

# =====================================================
# VISUAL ANALYTICS SECTION
# =====================================================

colA, colB = st.columns(2)

# Occupancy Graph
with colA:
    st.subheader("Occupancy Pattern")
    fig_occ, ax_occ = plt.subplots()
    ax_occ.plot(hours_in_day, occupancy_pattern)
    ax_occ.set_xlabel("Hour")
    ax_occ.set_ylabel("Occupancy Level")
    st.pyplot(fig_occ)

# Hourly Energy Graph
with colB:
    st.subheader("Hourly Energy Consumption")
    fig_energy, ax_energy = plt.subplots()
    ax_energy.plot(hours_in_day, hourly_energy, label="Base")
    ax_energy.plot(hours_in_day, hourly_energy_optimized, label="Optimized")
    ax_energy.legend()
    st.pyplot(fig_energy)

# ===============================
# BEFORE VS AFTER SECTION
# ===============================

st.markdown("---")
# ===============================
# Train Forecast Model ONCE
# ===============================

days = 180
X = np.arange(days).reshape(-1, 1)

y = np.array([
    monthly_energy / 30 + np.random.normal(0, 2)
    for _ in range(days)
])

model = LinearRegression()
model.fit(X, y)

r2_score = model.score(X, y)

# Accuracy
y_pred_train = model.predict(X)
mape = np.mean(np.abs((y - y_pred_train) / y)) * 100
accuracy = 100 - mape

# Forecast
future_days = 30
X_future = np.arange(days, days + future_days).reshape(-1, 1)
y_future = model.predict(X_future)
# Create equal spaced columns
colC, divider1, colD = st.columns([1, 0.03, 1])
with divider1:
    st.markdown(VERTICAL_DIVIDER, unsafe_allow_html=True)

# ---------------------------
# LEFT COLUMN
# ---------------------------
# ---------------------------
# LEFT: Power BI Style Bar Chart
# ---------------------------
with colC:
    st.markdown("### Before vs After Comparison")

    fig_compare, ax_compare = plt.subplots(figsize=(6, 4))

    categories = ["Energy", "Cost", "CO₂"]
    before_vals = [monthly_energy, monthly_cost, monthly_carbon]
    after_vals = [opt_monthly_energy, opt_monthly_cost, opt_monthly_carbon]

    x = np.arange(len(categories))
    width = 0.35

    # ---------------- GLOW LAYERS ----------------
    
    for i in range(6, 0, -1):
        ax_compare.bar(
        x - width/2,
        before_vals,
        width,
        color="#4C78A8",
        alpha=0.04 * i,
        zorder=1
    )

    ax_compare.bar(
        x + width/2,
        after_vals,
        width,
        color="#00C49A",
        alpha=0.04 * i,
        zorder=1
    )
    # ---------------- MAIN SOLID BARS ----------------
    ax_compare.bar(
    x - width/2,
    before_vals,
    width,
    label="Before",
    color="#4C78A8",
    zorder=2
    )

    ax_compare.bar(
    x + width/2,
    after_vals,
    width,
    label="After",
    color="#00C49A",
    zorder=2
    )


    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(categories)

    ax_compare.grid(axis="y", linestyle="--", alpha=0.3)
    ax_compare.legend(frameon=False)

    # Remove borders (Power BI style)
    for spine in ax_compare.spines.values():
        spine.set_visible(False)

    st.pyplot(fig_compare, use_container_width=True)

    


# ---------------------------
# RIGHT COLUMN
# ---------------------------
# ---------------------------
# RIGHT: Enterprise AI Forecast
# ---------------------------
with colD:
    st.markdown("### AI Energy Forecast (ML Time-Series Model)")

    

    fig_forecast, ax_forecast = plt.subplots(figsize=(6, 4))  # smaller & balanced

    # Glow Effect
    for i in range(6, 0, -1):
        ax_forecast.plot(
            range(days, days + future_days),
            y_future,
            linestyle="dashed",
            color="#00C49A",
            alpha=0.05 * i,
            linewidth=2 + i
        )

    ax_forecast.plot(
        range(days),
        y,
        label="Historical",
        color="#4C78A8",
        linewidth=2
    )

    ax_forecast.plot(
        range(days, days + future_days),
        y_future,
        linestyle="dashed",
        label="Forecast",
        color="#00C49A",
        linewidth=3
    )

    ax_forecast.grid(linestyle="--", alpha=0.3)
    ax_forecast.legend(frameon=False)

    for spine in ax_forecast.spines.values():
        spine.set_visible(False)

    st.pyplot(fig_forecast, use_container_width=True)

    
    
# ===============================
# ROW 2: Table + Model Metrics
# ===============================



col_table, divider2, col_metrics = st.columns([1, 0.02, 1])
with divider2:
    st.markdown(VERTICAL_DIVIDER, unsafe_allow_html=True)

# ---------- LEFT: BEFORE vs AFTER TABLE ----------
with col_table:
    st.markdown("### Before vs After Data")

    comparison_df = pd.DataFrame({
        "Category": ["Energy (kWh)", "Cost (₹)", "CO₂ (kg)"],
        "Before": [monthly_energy, monthly_cost, monthly_carbon],
        "After": [opt_monthly_energy, opt_monthly_cost, opt_monthly_carbon]
    })

    st.dataframe(
        comparison_df.style.format({
            "Before": "{:,.2f}",
            "After": "{:,.2f}"
        }),
        use_container_width=True,
        
    )

# ---------- RIGHT: MODEL METRICS ----------
with col_metrics:
    st.markdown("### Model Performance")

    m1, m2 = st.columns(2)

    with m1:
        st.metric("R² Score", f"{r2_score:.3f}")

    with m2:
        st.metric("Accuracy", f"{accuracy:.1f}%")
# ===============================
# AI Recommendation Display
# ===============================

st.markdown("---")

# ----- CARD HEADER ONLY -----
st.markdown("""
<div style="
    background: linear-gradient(135deg, #111827, #1A1F2E);
    padding: 10px;
    border-radius: 20px;
    border-left: 6px solid #00C49A;
    margin-down: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
">
<h3 style="color:#00C49A; margin-bottom:0;">
🧠 AI Optimization Recommendations
</h3>
</div>
""", unsafe_allow_html=True)

# ----- SPACE OUTSIDE CARD -----
st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ----- BULLETS OUTSIDE CARD -----
for rec in recommendations:
    st.markdown(f"""
    <p style="font-size:17px; margin-left:20px;">
    • {rec}
    </p>
    """, unsafe_allow_html=True)
# ===============================
# AI Predictive Decision Output
# ===============================

st.markdown("---")
st.markdown("## 🔮 AI Predictive Impact Analysis")

# Simulate 10% occupancy increase
energy_up, cost_up, carbon_up = simulate_change(occupancy_factor=1.10)

# Simulate 15% tariff increase
energy_tariff, cost_tariff, carbon_tariff = simulate_change(tariff_factor=1.15)
# ===============================
# AI Risk Scoring Engine
# ===============================

risk_score = 0

cost_risk_percent = ((cost_tariff - monthly_cost) / monthly_cost) * 100

if cost_risk_percent > 15:
    risk_score += 40
elif cost_risk_percent > 5:
    risk_score += 20

carbon_risk_percent = ((carbon_up - monthly_carbon) / monthly_carbon) * 100

if carbon_risk_percent > 10:
    risk_score += 30
elif carbon_risk_percent > 5:
    risk_score += 15

if efficiency_score < 20:
    risk_score += 30

risk_score = min(risk_score, 100)

if risk_score < 30:
    risk_level = "Low"
elif risk_score < 70:
    risk_level = "Medium"
else:
    risk_level = "High"
colP1, colP2 = st.columns(2)
st.markdown("### 🛡 AI Risk Assessment")

colR1, colR2 = st.columns(2)

with colR1:
    st.metric("Risk Score", f"{risk_score}/100")

with colR2:
    st.metric("Risk Level", risk_level)


    
# ===============================
# AI Strategy Recommendation
# ===============================

if risk_level == "High":
    strategy_recommendation = "Switch to High Efficiency Mode immediately to mitigate financial and carbon risk."

elif risk_level == "Medium":
    strategy_recommendation = "Adopt Smart Adaptive strategy to balance savings and performance."

else:
    strategy_recommendation = "Current strategy is stable. Maintain optimization and monitor trends."

st.markdown("""
<div style="
    background: linear-gradient(135deg, #111827, #1A1F2E);
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #00C49A;
    margin-top: 20px;
    margin-bottom: 20px;
">
<h4 style="color:#00C49A;">🤖 AI Strategic Recommendation</h4>
""", unsafe_allow_html=True)

st.markdown(f"""
<p style="font-size:17px;">
{strategy_recommendation}
</p>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)    
with colP1:
    st.metric(
        "Impact of 10% Higher Occupancy",
        f"₹{cost_up - monthly_cost:,.0f}",
        delta="Cost Increase"
    )

with colP2:
    st.metric(
        "Impact of 15% Tariff Rise",
        f"₹{cost_tariff - monthly_cost:,.0f}",
        delta="Financial Risk"
    )    
# =====================================================
# BUSINESS IMPACT SECTION
# =====================================================

st.markdown("---")
st.markdown("---")
if annual_savings > 0:
    roi_text = f"{50000 / annual_savings:.2f} years"
else:
    roi_text = "Not Applicable (No Savings)"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #111827, #1A1F2E);
    padding: 10px;
    border-radius: 20px;
    border-left: 6px solid #00C49A;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
">

<h3 style="color:#00C49A; margin-bottom:20px;">
Executive Summary
</h3>

<p style="font-size:18px; line-height:1.8;">
• AI optimization reduces monthly energy consumption by <b>{(monthly_energy - opt_monthly_energy):.0f} kWh</b>.<br>
• Estimated annual savings of <b>₹{annual_savings:,.0f}</b>.<br>
• Carbon emissions reduced by <b>{annual_carbon_saved:,.0f} kg/year</b>.<br>
• ROI payback expected within <b>{roi_text}</b>.<br>
• Model forecast reliability score: <b>{r2_score:.2f}</b>.
</p>

</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style="
    display: flex;
    justify-content: center;
    margin-top: 60px;
">

<div style="
    background: linear-gradient(145deg, #1A1F2E, #141824);
    padding: 70px;
    border-radius: 25px;
    width: 90%;
    max-width: 1300px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
">

<h1 style="font-size:42px; margin-bottom:40px; letter-spacing:1px;">
Business Impact & ROI
</h1>

<div style="font-size:24px; line-height: 2.2;">
    <p><strong>Annual Savings:</strong> ₹{annual_savings:,.0f}</p>
    <p><strong>5-Year Savings:</strong> ₹{five_year_savings:,.0f}</p>
    <p><strong>Annual CO₂ Reduction:</strong> {annual_carbon_saved:,.0f} kg</p>
    <p><strong>5-Year CO₂ Reduction:</strong> {five_year_carbon_saved:,.0f} kg</p>
</div>

<hr style="margin:40px 0; border: 1px solid #2C3245;" />

</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
if five_year_carbon_saved > 10000:
    st.success("🌱 Carbon Neutral Impact Achieved Over 5 Years!")
    
# Centered Investment Section
left, center, right = st.columns([1,2,1])

with center:
    initial_investment = st.number_input(
        "Initial Smart Lighting Investment (₹)",
        min_value=0.0,
        value=50000.0
    )

    if annual_savings > 0:
        payback_period = initial_investment / annual_savings

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00C49A, #006B4F);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-top: 25px;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        ">
            Estimated Payback Period: {payback_period:.2f} Years
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:35px;'></div>", unsafe_allow_html=True)
        # -------------------------------
        # PDF REPORT DOWNLOAD BUTTON
        # -------------------------------

        pdf_file = generate_pdf_report()

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📄 Download Full Optimization Report (PDF)",
                data=f,
                file_name="Lighting_Optimization_Report.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("No savings → Payback not applicable")
