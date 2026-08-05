import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score

# ==========================================
# 1. Page Configuration & Header
# ==========================================
st.set_page_config(page_title="Geospatial Prospectivity AI", layout="wide")
st.title("AI-Driven Mineral Prospectivity & Exploration Costing Dashboard")
st.caption("Multi-Source Geospatial ML for Orogenic Gold, SEDEX, and VMS Deposit Systems")

# ==========================================
# 2. Dynamic Dataset Generator
# ==========================================
@st.cache_data(show_spinner=False)
def load_geological_dataset(system_type, grid_size=100):
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Geostatistical spatial autocorrelation parameter calibrated for Kriging
    kriging_range = 25
    
    if system_type == "Orogenic Gold System":
        np.random.seed(101)
        sar_struct = np.exp(-((X-5)**2 + (Y-5)**2)/(kriging_range/8)) + np.random.normal(0, 0.08, (grid_size, grid_size))
        geochem_kriged = np.sin(X/2) * np.cos(Y/2) + np.random.normal(0, 0.12, (grid_size, grid_size))
        sentinel_alt = np.abs(np.sin(X) + np.cos(Y)) * 0.7 + np.random.normal(0, 0.05, (grid_size, grid_size))
        ground_truth = (sar_struct > 0.55) & (sentinel_alt > 0.45)
        
    elif system_type == "SEDEX Deposit System":
        np.random.seed(202)
        geochem_kriged = np.exp(-((X-3)**2 + (Y-7)**2)/(kriging_range/6)) + np.exp(-((X-7)**2 + (Y-3)**2)/(kriging_range/6)) + np.random.normal(0, 0.05, (grid_size, grid_size))
        sar_struct = np.sin(X) * 0.4 + np.random.normal(0, 0.1, (grid_size, grid_size))
        sentinel_alt = np.cos(Y) * 0.5 + np.random.normal(0, 0.1, (grid_size, grid_size))
        ground_truth = (geochem_kriged > 0.50) & (sar_struct > 0.15)
        
    else:  # VMS Hydrothermal System
        np.random.seed(303)
        sentinel_alt = np.exp(-((X-6)**2 + (Y-6)**2)/(kriging_range/10)) + np.random.normal(0, 0.05, (grid_size, grid_size))
        geochem_kriged = np.exp(-((X-6)**2 + (Y-6)**2)/(kriging_range/12)) + np.random.normal(0, 0.08, (grid_size, grid_size))
        sar_struct = np.abs(np.sin(X) * np.cos(Y)) + np.random.normal(0, 0.1, (grid_size, grid_size))
        ground_truth = (sentinel_alt > 0.45) & (geochem_kriged > 0.40)
        
    return geochem_kriged, sar_struct, sentinel_alt, ground_truth.astype(int)

# ==========================================
# 3. Sidebar: Configuration Panel
# ==========================================
st.sidebar.header("1. Target Deposit System")
selected_dataset = st.sidebar.selectbox(
    "Select Geological System Dataset",
    ("Orogenic Gold System", "SEDEX Deposit System", "VMS Hydrothermal System")
)

st.sidebar.markdown("---")
st.sidebar.header("2. ML Algorithm & Settings")
model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ("Multi-Layer Perceptron (MLP)", "Support Vector Machine (SVM)")
)

decision_threshold = st.sidebar.slider(
    "Prospectivity Probability Threshold", 0.10, 0.90, 0.50, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.header("3. Spatial Layer Weights")
w_geochem = st.sidebar.slider("Geochemistry Weight (Kriged Range=25)", 0.0, 2.0, 1.0, 0.1)
w_sar = st.sidebar.slider("SAR Lineaments Weight (Structure)", 0.0, 2.0, 1.0, 0.1)
w_sentinel = st.sidebar.slider("Sentinel-2 Alteration Weight", 0.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("4. Expedition Financial Parameters")
cost_per_drill = st.sidebar.number_input("Field Drilling Cost per Target Cell ($)", value=15000, step=1000)
penalty_fp = st.sidebar.number_input("Wasted Drilling Penalty per False Positive ($)", value=25000, step=2500)
penalty_fn = st.sidebar.number_input("Missed Deposit Opportunity Loss per False Negative ($)", value=75000, step=5000)

# ==========================================
# 4. Data Processing & Model Pipeline
# ==========================================
geochem, sar, sentinel, y_true_2d = load_geological_dataset(selected_dataset)
y_true = y_true_2d.flatten()

# Feature Matrix Generation
X_raw = np.column_stack((
    geochem.flatten() * w_geochem,
    sar.flatten() * w_sar,
    sentinel.flatten() * w_sentinel
))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Model Training
if model_choice == "Multi-Layer Perceptron (MLP)":
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
else:
    model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)

model.fit(X_scaled, y_true)
probs = model.predict_proba(X_scaled)[:, 1]

# Predictions based on user decision threshold
y_pred = (probs >= decision_threshold).astype(int)

prospectivity_map = probs.reshape(100, 100)
uncertainty_map = (1 - np.abs(2 * probs - 1)).reshape(100, 100)
target_binary_map = y_pred.reshape(100, 100)

# ==========================================
# 5. Statistical Error & Financial Metrics
# ==========================================
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
auc_score = roc_auc_score(y_true, probs)

# Financial Calculations
total_targets = tp + fp
drilling_budget = total_targets * cost_per_drill
wasted_fp_cost = fp * penalty_fp
missed_opportunity_cost = fn * penalty_fn
total_expedition_risk_cost = drilling_budget + wasted_fp_cost + missed_opportunity_cost

# ==========================================
# 6. Dashboard Display Section
# ==========================================

# Financial & Error Probability Summary Cards
st.subheader("Field Expedition Financials & Statistical Errors")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

m_col1.metric("False Positive Prob (FPR)", f"{fpr:.2%}", help="Probability of targeting a non-mineralized cell (Type I Error)")
m_col2.metric("False Negative Prob (FNR)", f"{fnr:.2%}", help="Probability of missing a real deposit cell (Type II Error)")
m_col3.metric("Model ROC-AUC", f"{auc_score:.3f}")
m_col4.metric("Total Drilling Budget", f"${drilling_budget:,.0f}")
m_col5.metric("Total Expedition Risk Capital", f"${total_expedition_risk_cost:,.0f}", delta=f"-${wasted_fp_cost:,.0f} Wasted", delta_color="inverse")

st.markdown("---")

# Dynamic Visualizations
st.subheader(f"Geospatial Analytics for {selected_dataset}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Dominant Evidence Layer**")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    if selected_dataset == "Orogenic Gold System":
        im1 = ax1.imshow(sar, cmap='magma')
        ax1.set_title("SAR Structural Lineaments")
    elif selected_dataset == "SEDEX Deposit System":
        im1 = ax1.imshow(geochem, cmap='viridis')
        ax1.set_title("Kriged Geochemistry (Range=25)")
    else:
        im1 = ax1.imshow(sentinel, cmap='plasma')
        ax1.set_title("Sentinel-2 Hydrothermal Alteration")
    fig1.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.axis('off')
    st.pyplot(fig1)

with col2:
    st.markdown("**Continuous Probability Map**")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    im2 = ax2.imshow(prospectivity_map, cmap='jet')
    ax2.set_title(f"{model_choice.split()[0]} Probability")
    fig2.colorbar(im2, ax=ax2, shrink=0.7)
    ax2.axis('off')
    st.pyplot(fig2)

with col3:
    st.markdown(f"**Target Zones (Threshold ≥ {decision_threshold:.2f})**")
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    im3 = ax3.imshow(target_binary_map, cmap='YlOrRd')
    ax3.set_title("Targeted Exploration Cells")
    fig3.colorbar(im3, ax=ax3, shrink=0.7)
    ax3.axis('off')
    st.pyplot(fig3)

with col4:
    st.markdown("**Model Uncertainty Map**")
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    im4 = ax4.imshow(uncertainty_map, cmap='bone')
    ax4.set_title("Prediction Uncertainty")
    fig4.colorbar(im4, ax=ax4, shrink=0.7)
    ax4.axis('off')
    st.pyplot(fig4)

st.markdown("---")

# Detailed Breakdown Table
col_tbl1, col_tbl2 = st.columns(2)

with col_tbl1:
    st.subheader("Confusion Matrix Analysis")
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=["Pred Barren (0)", "Pred Target (1)"],
        index=["Actual Barren (0)", "Actual Deposit (1)"]
    )
    st.dataframe(cm_df, use_container_width=True)

with col_tbl2:
    st.subheader("Expedition Cost Breakdown")
    cost_data = {
        "Cost Category": [
            "Planned Drilling Cost (TP + FP)",
            "Wasted Drilling Capital on Duds (FP)",
            "Unrealized Deposit Opportunity Loss (FN)",
            "Total Financial Risk"
        ],
        "Amount ($)": [
            f"${drilling_budget:,.2f}",
            f"${wasted_fp_cost:,.2f}",
            f"${missed_opportunity_cost:,.2f}",
            f"${total_expedition_risk_cost:,.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(cost_data), use_container_width=True)