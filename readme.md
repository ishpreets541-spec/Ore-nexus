# 🌍 Ore-Nexus
**Enterprise-Grade Geospatial Machine Learning for Economic Geology & Exploration Costing**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Database-336791?logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Data-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Ore-Nexus** is an interactive, multi-source predictive modeling dashboard designed to identify high-potential mineral deposit zones while simultaneously calculating expedition financial risk. By fusing advanced machine learning with spatial geostatistics, this tool bridges the gap between deep earth sciences and actionable exploration budgeting.

---

## 🚀 Live Demo
Access the live deployment on Streamlit Community Cloud: **https://ore-nexus.streamlit.app/**



---

## 🧠 Core Features

*   **Multi-Source Spatial Integration:** Fuses Sentinel-2 hydrothermal alteration halos, Synthetic Aperture Radar (SAR) structural backscatter lineaments, and geochemical point data. 
*   **Precision Geostatistics:** Geochemical interpolation utilizes strictly calibrated Ordinary Kriging, maintaining a fixed spatial autocorrelation semivariogram range of 25 for optimal regional stability.
*   **Advanced Algorithmic Targeting:** Deploys Multi-Layer Perceptrons (MLP), Support Vector Machines (SVM), and Random Forest ensemble architectures to delineate complex, non-linear geological boundaries.
*   **Financial Risk Engine:** Translates statistical classification errors directly into capital risk. Calculates wasted drilling capital from False Positives and massive unrealized opportunity costs from False Negatives.
*   **Explainable AI (XAI):** Features dynamic ROC-AUC curves and Random Forest feature importance charts to ensure geologists understand which spatial layer drove the model's predictions.

---

## 🪨 Modeled Deposit Systems

The system evaluates prospectivity signatures across three distinct geological environments:
1.  **Orogenic Gold Systems:** High weighting on structurally controlled SAR lineaments and secondary metamorphic alterations.
2.  **SEDEX (Sedimentary Exhalative) Deposits:** Driven by specific basin-scale geochemical anomalies and structural traps.
3.  **VMS (Volcanogenic Massive Sulfide):** Balanced hydrothermal mapping via Sentinel-2 and localized geochemical halos.

---

## 📊 Statistical & Financial Mathematics

The financial risk engine calculates the total expedition risk based on model accuracy, relying on the following core metrics:

*   **False Positive Rate (Wasted Capital):** $FPR = \frac{FP}{FP + TN}$
*   **False Negative Rate (Missed Opportunity):** $FNR = \frac{FN}{FN + TP}$

*Minimizing $FPR$ saves immediate drilling budget, while minimizing $FNR$ prevents the loss of billion-dollar undiscovered assets.*

---

## 💻 Local Installation & Setup

To run Ore-Nexus on your local machine for development or spatial layer modification:

---
### 👩‍💻 Author
**Ishpreet Singh**

M.Tech
Indian Institute of Technology Bombay
Mail ID:
25m0326@iitb.ac.in
---
