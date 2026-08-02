import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


# Page Configuration
st.set_page_config(
    page_title="ThermalBatteryPredictor",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: #667eea;
        color: white;
        padding: 1.2rem 2rem 0.8rem 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .stButton button {
        background: #1f77b4;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .feature-table {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stNumberInput input {
        font-family: monospace;
    }
    .material-info {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .comparison-section {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 2px solid #1f77b4;
    }
    .recommend-name {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .card-metrics {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 0.8rem;
    }
    .metric-item {
        flex: 1;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .figure-title {
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
    }
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .indent-title {
        padding-left: 10ch;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🔋 ThermalBatteryPredictor</div>', unsafe_allow_html=True)

# Session State Initialization
if 'heat_exchanger_result' not in st.session_state:
    st.session_state.heat_exchanger_result = None
if 'calc_value' not in st.session_state:
    st.session_state.calc_value = None
if 'comparison_needed' not in st.session_state:
    st.session_state.comparison_needed = False
if 'comparison_type' not in st.session_state:
    st.session_state.comparison_type = None
if 'show_input_prompt' not in st.session_state:
    st.session_state.show_input_prompt = False
if 'feedback_input' not in st.session_state:
    st.session_state.feedback_input = ""
if 'feedback_sent' not in st.session_state:
    st.session_state.feedback_sent = False


# GitHub Repository Configuration
GITHUB_USERNAME = "JJJ069"
GITHUB_REPO = "predictor"
GITHUB_BRANCH = "main"

MODEL_FILE_PATH = "model.pkl"
SCALER_FILE_PATH = "scaler.pkl"

MODEL_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FILE_PATH}"
SCALER_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{SCALER_FILE_PATH}"


# Sidebar
with st.sidebar:
    st.header("🔧 Model Settings")
    st.subheader("GitHub Configuration")
    github_username = st.text_input("GitHub Username", value=GITHUB_USERNAME)
    github_repo = st.text_input("Repository Name", value=GITHUB_REPO)
    github_branch = st.text_input("Branch", value=GITHUB_BRANCH)

    MODEL_URL = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/{github_branch}/{MODEL_FILE_PATH}"
    SCALER_URL = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/{github_branch}/{SCALER_FILE_PATH}"

    st.info(f"Model URL: [Link]({MODEL_URL})")
    st.info(f"Scaler URL: [Link]({SCALER_URL})")



# PCM Material Database
pcm_materials = {
    "Erythritol": {"LH": 333.7, "Density": 1346, "CP": 1.98},
    "MgCl2.6H2O": {"LH": 169, "Density": 1450, "CP": 1.83},
    "Xylitol": {"LH": 237.6, "Density": 1345, "CP": 1.27},
    "Ba(OH)2.8H2O": {"LH": 244.4, "Density": 1937, "CP": 2.9},
    "PA/Cu": {"LH": 174, "Density": 1348, "CP": 2.06},
    "RT60": {"LH": 186.7, "Density": 897, "CP": 2.17},
    "SAT/EG": {"LH": 227.3, "Density": 1000, "CP": 3.22},
    "Stearic acid": {"LH": 196.1, "Density": 900, "CP": 2.48},
    "Paraffin/EG": {"LH": 155, "Density": 300, "CP": 1.7},
    "Lauric acid": {"LH": 187.21, "Density": 912, "CP": 2.29},
    "Eicosane": {"LH": 248, "Density": 816, "CP": 2.16},
    "RT35/SiC": {"LH": 145, "Density": 917, "CP": 1.9},
    "Beeswax": {"LH": 242.8, "Density": 970, "CP": 0.476},
    "OP44E": {"LH": 233.5, "Density": 818, "CP": 2.15},
    "D-Mannitol": {"LH": 299, "Density": 1390, "CP": 2.5},
    "M15 mortar": {"LH": 183, "Density": 1710, "CP": 1.05},
    "CrodaTherm32": {"LH": 176.4, "Density": 880, "CP": 1.85},
    "Crodatherm74": {"LH": 190, "Density": 880, "CP": 1.85},
    "Paraffin wax-27": {"LH": 231.2, "Density": 808, "CP": 2.23},
    "Paraffin wax-50": {"LH": 114.54, "Density": 820, "CP": 2},
    "Paraffin wax-52": {"LH": 203.56, "Density": 826.24, "CP": 2.024},
    "Paraffin wax-55": {"LH": 102, "Density": 785, "CP": 2.85},
    "Paraffin wax-58": {"LH": 250, "Density": 800, "CP": 2.1},
    "Paraffin wax-60": {"LH": 190, "Density": 850, "CP": 2.08},
    "Paraffin wax-61": {"LH": 190, "Density": 850, "CP": 2.075},
    "Paraffin wax-63": {"LH": 218, "Density": 800, "CP": 2},
    "Paraffin RT18": {"LH": 209, "Density": 780, "CP": 2.16},
    "Paraffin RT23": {"LH": 195.2, "Density": 870, "CP": 2.12},
    "Paraffin RT25": {"LH": 170, "Density": 820, "CP": 2},
    "Paraffin RT27": {"LH": 179, "Density": 815, "CP": 2.1},
    "Paraffin RT28": {"LH": 250, "Density": 825, "CP": 2},
    "Paraffin RT35": {"LH": 220, "Density": 805, "CP": 3.55},
    "Paraffin RT42": {"LH": 165, "Density": 880, "CP": 2},
    "Paraffin RT44HC": {"LH": 253, "Density": 750, "CP": 2},
    "Paraffin RT45": {"LH": 195, "Density": 880, "CP": 2.1},
    "Paraffin RT50": {"LH": 168, "Density": 780, "CP": 2},
    "Paraffin RT52": {"LH": 115.682, "Density": 780, "CP": 2},
    "Paraffin RT55": {"LH": 170, "Density": 770, "CP": 2},
    "Paraffin RT58": {"LH": 230, "Density": 812, "CP": 0.9},
    "Paraffin RT64HC": {"LH": 220, "Density": 780, "CP": 2},
    "Paraffin RT65": {"LH": 150, "Density": 770, "CP": 2},
    "Paraffin RT82": {"LH": 176, "Density": 860, "CP": 2},
    "n-eicosane": {"LH": 250, "Density": 816, "CP": 2.39},
    "n-pentosane": {"LH": 238, "Density": 880, "CP": 2.52},
    "n-octadecane": {"LH": 245, "Density": 860, "CP": 2.065},
    "n-Hexadecane": {"LH": 236.99, "Density": 825, "CP": 2.45},
    "Tetradecanol": {"LH": 227.8, "Density": 835, "CP": 2.2},
    "Tetradecane": {"LH": 226, "Density": 798, "CP": 1.95},
    "Acetic acid": {"LH": 192, "Density": 1214, "CP": 1.67},
    "Myristic acid": {"LH": 252.92, "Density": 855.5, "CP": 2.05},
    "Hexadecanoic acid": {"LH": 209.19, "Density": 853, "CP": 1.804},
    "Pb": {"LH": 26, "Density": 11340, "CP": 0.13},
    "Mg": {"LH": 146, "Density": 2260, "CP": 1.02},
    "Zn": {"LH": 112, "Density": 7140, "CP": 0.44},
    "Gallium": {"LH": 80.16, "Density": 6093, "CP": 0.3815},
    "CH3COONa·3H2O": {"LH": 264, "Density": 1341, "CP": 2.93},
    "CaCl2.6H2O": {"LH": 170, "Density": 1622, "CP": 2.15},
    "Na2SO4.10H2O": {"LH": 180, "Density": 1485, "CP": 2.37},
    "NaNO2": {"LH": 212, "Density": 2170, "CP": 0.25},
    "NaNO3": {"LH": 178, "Density": 2019, "CP": 0.45},
    "LiNO3": {"LH": 373, "Density": 2380, "CP": 0.93},
    "NaOH": {"LH": 373, "Density": 2380, "CP": 0.93},
    "KOH": {"LH": 149, "Density": 2044, "CP": 1.47},
    "ZnCl2": {"LH": 75, "Density": 2907, "CP": 0.74},
    "MgCl2": {"LH": 452, "Density": 2330, "CP": 0.97},
    "KCl": {"LH": 353, "Density": 2330, "CP": 0.97},
    "KNO3+NaNO3": {"LH": 94, "Density": 1705, "CP": 1.42},
    "NaNO3+NaOH": {"LH": 292, "Density": 2030, "CP": 1.89},
    "NaNO3+KNO3+NaNO2": {"LH": 80, "Density": 1976.6, "CP": 1.465},
    "KNO3+LiNO3+NaNO2": {"LH": 89.68, "Density": 2319, "CP": 1.776},
    "Sodium nitrate and potassium": {"LH": 108, "Density": 2010, "CP": 1.424},
    "Molten salts": {"LH": 110, "Density": 1994, "CP": 1.626},
    "Custom": {"LH": 0, "Density": 0, "CP": 0}
}


# Section 1: Inverse Selection of PCM Heat Exchanger
st.markdown("### 1. Inverse selection of PCM heat exchanger")

with st.expander("PCM heat exchanger selection", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        Q = st.number_input(
            "Heat storage capacity (Q, kW·h)",
            min_value=0.0,
            max_value=1000.0,
            value=14.5,
            step=0.1,
            help="Total heat storage capacity of the thermal battery"
        )
        material_options = list(pcm_materials.keys())
        selected_material = st.selectbox(
            "Select phase change material (PCM)",
            options=material_options,
            index=0,
            help="Select PCM material from the database"
        )

        if selected_material == "Custom":
            custom_lh = st.number_input(
                "Latent heat (kJ/kg)",
                min_value=0.0,
                max_value=500.0,
                value=333.7,
                step=0.1
            )
            custom_density = st.number_input(
                "Density (kg/m³)",
                min_value=0.0,
                max_value=3000.0,
                value=1346.0,
                step=0.1
            )
            custom_cp = st.number_input(
                "Specific heat capacity (kJ/kg·K)",
                min_value=0.0,
                max_value=10.0,
                value=1.98,
                step=0.01
            )
            LH_value = custom_lh
            Density_value = custom_density
            CP_value = custom_cp
        else:
            material_data = pcm_materials[selected_material]
            LH_value = material_data["LH"]
            Density_value = material_data["Density"]
            CP_value = material_data["CP"]

            st.markdown('<div class="material-info">', unsafe_allow_html=True)
            st.markdown(f"**Material properties:**")
            st.markdown(f"- Latent heat: {LH_value:.2f} kJ/kg")
            st.markdown(f"- Density: {Density_value:.2f} kg/m³")
            st.markdown(f"- Specific heat capacity: {CP_value:.2f} kJ/(kg·K)")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        Ti = st.number_input(
            "Initial temperature T$_{\mathrm{i}}$ (°C)",
            min_value=-50.0,
            max_value=500.0,
            value=50.0,
            step=0.1,
            help="Initial temperature of the thermal battery"
        )
        Th = st.number_input(
            "Maximum temperature T$_{\mathrm{h}}$ (°C)",
            min_value=0.0,
            max_value=500.0,
            value=150.0,
            step=0.1,
            help="Maximum operating temperature"
        )
        HTA_hex = st.number_input(
            "Estimated heat transfer area (HTA, m²)",
            min_value=0.0,
            max_value=100.0,
            value=0.5655,
            step=0.0001,
            format="%.4f",
            help="Estimated heat transfer surface area"
        )

    if st.button("🔍 Recommended PCM heat exchanger", use_container_width=True):
        Q_kJ = Q * 3600
        delta_T = Th - Ti

        if delta_T <= 0:
            st.error("error: Th must be higher than T$_{\mathrm{i}}$")
        else:
            calc_value = HTA_hex / ((Q_kJ / (CP_value * delta_T + LH_value)) / Density_value)
            st.session_state.calc_value = calc_value

            st.markdown(f"** Heat transfer surface area density (HTA/Vpcm, m²/m³) = {calc_value:.2f}")

            if calc_value >= 300:
                st.session_state.heat_exchanger_result = "Flat-plate heat exchanger"
                st.session_state.comparison_needed = False
                st.session_state.show_input_prompt = True
                st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Flat-plate heat exchanger</span>  
**Because heat transfer surface area density (HTA/Vpcm, m²/m³) ≥ 300**
""", unsafe_allow_html=True)

            elif 200 < calc_value < 300:
                st.session_state.comparison_needed = True
                st.session_state.comparison_type = "two_type"
                st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
                st.markdown("### ⚖️ Further comparison: Flat-plate vs shell-and-tube")
                st.markdown("**Reason:** 200 < Heat transfer surface area density (HTA/Vpcm, m²/m³) < 300")
                st.markdown("Please input the following parameters and click 'Compare heat exchangers'")
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.session_state.comparison_needed = True
                st.session_state.comparison_type = "three_type"
                st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
                st.markdown("### ⚖️ Further comparison: Three types")
                st.markdown("**Reason:** Heat transfer surface area density (HTA/Vpcm, m²/m³) ≤ 200")
                st.markdown("Please input the following parameters and click 'Compare heat exchangers'")
                st.markdown('</div>', unsafe_allow_html=True)

# Show comparison inputs if needed
if st.session_state.comparison_needed:
    st.markdown("### 📐 Parameter comparisons")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        W = st.number_input(
            "PCM width for Flat-plate (W, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.03,
            step=0.001,
            format="%.4f",
            key="W_input"
        )

    with col_b:
        di = st.number_input(
            "Shell-and-tube inner diameter (d$_{\mathrm{i}}$, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.018,
            step=0.001,
            format="%.4f",
            key="di_input"
        )

    with col_c:
        do = st.number_input(
            "Shell-and-tube outer diameter (d$_{\mathrm{o}}$, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.03,
            step=0.001,
            format="%.4f",
            key="do_input"
        )

    if st.session_state.comparison_type == "three_type":
        D = st.number_input(
            "Spherical capsule outer diameter (D, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.4f",
            key="D_input"
        )

    if st.button("🔍 Compare heat exchangers", key="compare_button"):
        if st.session_state.comparison_type == "two_type":
            comparison_value = 4 * W * di / (do ** 2 - di ** 2)
            st.markdown(f"**Ratio of heat transfer surface area density: $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ = {comparison_value:.4f}")

            if comparison_value >= 1:
                st.session_state.heat_exchanger_result = "Shell-and-tube heat exchanger"
                st.session_state.show_input_prompt = True
                st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Shell-and-tube heat exchanger</span>  
**Because the ratio of heat transfer surface area density:** $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ ≥ 1
""", unsafe_allow_html=True)
            else:
                st.session_state.heat_exchanger_result = "Flat-plate heat exchanger"
                st.session_state.show_input_prompt = True
                st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Flat-plate heat exchanger</span>  
**Because the ratio of heat transfer surface area density:** $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ < 1
""", unsafe_allow_html=True)

        elif st.session_state.comparison_type == "three_type":
            comparison_value1 = 2 * D * di / (3 * (do ** 2 - di ** 2))
            st.markdown(f"**Ratio of heat transfer surface area density: $\\frac{{2Dd_i}}{{3(d_o^2 - d_i^2)}}$ = {comparison_value1:.4f}")

            if comparison_value1 >= 1:
                st.session_state.heat_exchanger_result = "Shell-and-tube heat exchanger"
                st.session_state.show_input_prompt = True
                st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Shell-and-tube heat exchanger</span>  
**Because the ratio of heat transfer surface area density:** $\\frac{{2Dd_i}}{{3(d_o^2 - d_i^2)}}$ ≥ 1
""", unsafe_allow_html=True)
            else:
                comparison_value2 = D / (6 * W)
                st.markdown(f"**D/6W:** {comparison_value2:.6f}")

                if comparison_value2 > 1:
                    st.session_state.heat_exchanger_result = "Flat-plate heat exchanger"
                    st.session_state.show_input_prompt = True
                    st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Flat-plate heat exchanger</span>  
**Because the ratio of heat transfer surface area density:** D/6W > 1
""", unsafe_allow_html=True)
                else:
                    st.session_state.heat_exchanger_result = "Spherical capsule heat exchanger"
                    st.session_state.show_input_prompt = True
                    st.markdown(f"""
**✅ Recommended:** <span class="recommend-name">Spherical capsule heat exchanger</span>  
**Because the ratio of heat transfer surface area density:** D/6W ≤ 1
""", unsafe_allow_html=True)

# Display final recommendation if available
if st.session_state.heat_exchanger_result and not st.session_state.comparison_needed:
    st.markdown(f"**Final recommendation:** {st.session_state.heat_exchanger_result}")

# Show instruction after heat exchanger selection
if st.session_state.show_input_prompt:
    st.info(
        "After selecting the heat exchanger, please click “Start” to predict the power density")


# Section 2: Forward Prediction of Power Density
# Feature configuration (keys must match FEATURE_ORDER)
feature_config = {
    'HCT(h)': {
        'options': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'default': 5.0,
        'description': 'Heat charging time'
    },
    'LH(kJ/kg)': {
        'options': [150, 200, 250, 300, 333.7, 350],
        'default': 333.7,
        'description': 'Latent heat'
    },
    'TC(W/mK)': {
        'options': [0.2, 0.4, 0.6, 0.8],
        'default': 0.8,
        'description': 'Thermal conductivity'
    },
    'Density(kg/m3)': {
        'options': [300, 600, 900, 1200, 1346, 1500],
        'default': 1346,
        'description': 'PCM density'
    },
    'CP(kJ/kgK)': {
        'options': [1.7, 1.8, 1.9, 1.98, 2.0],
        'default': 1.98,
        'description': 'Specific heat capacity'
    },
    'Mass(kg)': {
        'options': [30, 60],
        'default': 60,
        'description': 'Mass'
    },
    'FVR': {
        'options': [0.01, 0.02, 0.0245, 0.03, 0.04],
        'default': 0.0245,
        'description': 'Fin volume ratio'
    },
    'ES(m2)': {
        'options': [3, 5.4358, 6],
        'default': 5.4358,
        'description': 'Extended surface'
    },
    'FFL(m)': {
        'options': [0.2, 0.8, 1.4, 2.0],
        'default': 2.0,
        'description': 'Fluid flow length'
    },
    'HTA(m2)': {
        'options': [0.2, 0.565487, 0.6],
        'default': 0.565487,
        'description': 'Heat transfer area'
    },
    'TH(°C)': {
        'options': [10, 20, 30, 31.2, 40],
        'default': 31.2,
        'description': 'Degree of superheat'
    },
    'TP(°C)': {
        'options': [5, 10, 15, 18.8, 20, 25, 30],
        'default': 18.8,
        'description': 'Degree of subcooling'
    },
    'LPH(L/h)': {
        'options': [100, 200, 300, 400],
        'default': 200,
        'description': 'Litres per hour'
    }
}

# Define the exact order of features expected by the model
FEATURE_ORDER = [
    'HCT(h)', 'LH(kJ/kg)', 'TC(W/mK)', 'Density(kg/m3)', 'CP(kJ/kgK)', 'Mass(kg)',
    'FVR', 'ES(m2)', 'FFL(m)', 'HTA(m2)', 'TH(°C)', 'TP(°C)', 'LPH(L/h)'
]


# Model and Scaler Loading Functions
@st.cache_resource
def download_from_github(url, file_type="model"):
    """Download a file from GitHub and load it as model or scaler."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_content = io.BytesIO(response.content)
        if file_type == "model":
            model = joblib.load(file_content)
            st.sidebar.success(f"✅ Model loaded successfully from GitHub!")
            return model
        elif file_type == "scaler":
            scaler = joblib.load(file_content)
            st.sidebar.success(f"✅ Scaler loaded successfully from GitHub!")
            return scaler
        else:
            return file_content
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"❌ Failed to download {file_type}: {str(e)}")
        st.sidebar.info(f"Please check the URL: {url}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load {file_type}: {str(e)}")
        return None

@st.cache_resource
def load_resources():
    """Load the model and scaler from GitHub."""
    model = download_from_github(MODEL_URL, "model")
    scaler = download_from_github(SCALER_URL, "scaler")
    return model, scaler

model, scaler = load_resources()

def format_value(value, decimals=6):
    """Format a number to a fixed number of decimal places."""
    return float(f"{value:.{decimals}f}")

def preprocess_input(input_features, scaler):
    """Apply the same preprocessing (standardization) as used during training."""
    try:
        ordered_features = {feature: input_features[feature] for feature in FEATURE_ORDER}
        input_df = pd.DataFrame([ordered_features])
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_df_scaled = pd.DataFrame(input_scaled, columns=FEATURE_ORDER)
            return input_df_scaled
        else:
            return input_df
    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        return None

# Function to load image from URL safely
@st.cache_data(show_spinner=False)
def load_image_from_url(url):
    """Download image from URL and return PIL Image object."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        return None

# Titles for the first row of small figures (1.2-1.5) - for power density
SMALL_FIG_TITLES_PD = {
    "1.2": "Power density prediction model",
    "1.3": "Feature variable importance levels for power density",
    "1.4": "SHAP summary plot for power density",
    "1.5": "Cumulative probability plot for power density",
}

# Titles for the second row of small figures (2.2-2.5) - for SOC
SMALL_FIG_TITLES_SOC = {
    "2.2": "SOC prediction model",
    "2.3": "Feature variable importance levels for SOC",
    "2.4": "SHAP summary plot for SOC",
    "2.5": "Cumulative probability plot for SOC",
}


# Main Prediction Interface
if model is not None and scaler is not None:
    st.sidebar.success("✅ All resources loaded successfully!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 2. Forward prediction of power density")

    # Use 3 columns for input layout
    cols = st.columns(3)
    input_features = {}

    # Distribute features evenly among columns
    features_per_col = (len(FEATURE_ORDER) + 2) // 3
    for i, feature in enumerate(FEATURE_ORDER):
        col_idx = i // features_per_col
        if col_idx >= len(cols):
            col_idx = len(cols) - 1
        with cols[col_idx]:
            config = feature_config[feature]
            default_index = config['options'].index(config['default'])
            input_value = st.selectbox(
                label=f"{feature}: {config['description']}",
                options=config['options'],
                index=default_index,
                key=f"feature_{feature}",
            )
            input_features[feature] = format_value(input_value, 6)

    # Display input parameter table
    st.markdown("### 📋 Input parameter overview")
    formatted_values = [format_value(val, 6) for val in input_features.values()]
    param_index = list(range(1, len(input_features) + 1))
    features_display_df = pd.DataFrame({
        'No.': param_index,
        'Parameter name': list(input_features.keys()),
        'Parameter value': formatted_values,
        'Parameter description': [feature_config[name]['description'] for name in input_features.keys()]
    })
    pd.options.display.float_format = '{:.6f}'.format
    st.dataframe(features_display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Start Button
    col_btn, _ = st.columns([6, 4])
    with col_btn:
        start_clicked = st.button("🚀 Start", use_container_width=True)

    if start_clicked:
        with st.spinner("Calculating the predicted value..."):
            try:
                processed_data = preprocess_input(input_features, scaler)
                if processed_data is not None:
                    prediction = model.predict(processed_data)[0]
                    hct_val = input_features['HCT(h)']
                    energy_density = prediction * hct_val

                    # Row 1: Prediction card (60%) + Figures (40%)
                    col_card, col_fig1 = st.columns([6, 4])

                    with col_card:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2 style="margin-top: 0.2rem;">📈 Predicted value</h2>
                            <h1 style="font-size: 3rem; margin: 0.5rem 0;">{prediction:.2f} ± 3.4% kW/m<sup>3</sup></h1>
                            <p>According to {len(input_features)} thermodynamic parameters to predict power density</p>
                            <p style="font-size:1rem; margin-top:0.3rem;"> ±3.4% (mean absolute percentage error based on test dataset)</p>
                            <div class="card-metrics">
                                <div class="metric-item">
                                    <div class="metric-label">Predicted power density</div>
                                    <div class="metric-value">{prediction:.2f} kW/m³</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-label">Predicted state of charge ( SOC = PD * t )</div>
                                    <div class="metric-value">{energy_density:.2f} kWh/m³</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("✅ Data preprocessing completed (standardization applied)")

                    # Figure 1.1 and Figure 2.1 side by side
                    with col_fig1:
                        col_fig1_left, col_fig1_right = st.columns(2)

                        with col_fig1_left:
                            st.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: flex-start;">
                                <div style="padding-left: 8ch; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.2rem; text-align: left;">Variable correlation heatmap for power density</div>
                            """, unsafe_allow_html=True)
                            url_11 = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/1.1.png"
                            img = load_image_from_url(url_11)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.error("Failed to load Figure 1.1. Please check the file.")
                            st.markdown("</div>", unsafe_allow_html=True)

                        with col_fig1_right:
                            st.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: flex-start;">
                                <div style="padding-left: 10ch; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.2rem; text-align: left;">Variable correlation heatmap for SOC</div>
                            </div>
                            """, unsafe_allow_html=True)
                            url_21 = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/2.1.png"
                            img = load_image_from_url(url_21)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.error("Failed to load Figure 2.1. Please check the file.")
                            st.markdown("</div>", unsafe_allow_html=True)

                    # Row 3: Four small figures (1.2-1.5)
                    st.markdown("---")
                    cols_small = st.columns(4)
                    for idx, fig_key in enumerate(["1.2", "1.3", "1.4", "1.5"]):
                        with cols_small[idx]:
                            title_text = SMALL_FIG_TITLES_PD.get(fig_key, f"Figure {fig_key}")
                            st.markdown(f"""
                            <div class="center-container">
                                <div class="figure-title">{title_text}</div>
                            """, unsafe_allow_html=True)
                            url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{fig_key}.png"
                            img = load_image_from_url(url)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.error(f"Failed to load Figure {fig_key}. Please check the file.")
                            st.markdown("</div>", unsafe_allow_html=True)

                    # Second row: Four small figures (2.2-2.5) for SOC
                    st.markdown("---")
                    cols_small_soc = st.columns(4)
                    for idx, fig_key in enumerate(["2.2", "2.3", "2.4", "2.5"]):
                        with cols_small_soc[idx]:
                            title_text = SMALL_FIG_TITLES_SOC.get(fig_key, f"Figure {fig_key}")
                            st.markdown(f"""
                            <div class="center-container">
                                <div class="figure-title">{title_text}</div>
                            """, unsafe_allow_html=True)
                            url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{fig_key}.png"
                            img = load_image_from_url(url)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.error(f"Failed to load Figure {fig_key}. Please check the file.")
                            st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Errors occurred during the predicting process: {str(e)}")

else:
    # Model loading failed – display help
    st.markdown("""
    ## ⚠️ Unable to load model

    Please check the following:

    1. **GitHub Configuration** - Ensure the GitHub username, repository name, and branch are correct in the sidebar
    2. **File Names** - Make sure the model files are named correctly:
       - `model.pkl` for the model
       - `scaler.pkl` for the scaler
    3. **File Locations** - Ensure files are in the root directory of your GitHub repository
    4. **File Accessibility** - Check that the repository is public or you have access if it's private

    ### ⚙️ Configuration Instructions:

    Update the following in the sidebar:
    - **GitHub Username**: Your GitHub username
    - **Repository Name**: The name of your repository containing the model files
    - **Branch**: Usually "main" or "master"

    ### 📋 System parameters description：

    - **HCT(h)**: Heat charging time    
    - **LH(kJ/kg)**: Latent heat
    - **TC(W/mK)**: Thermal conductivity
    - **Density(kg/m3)**: PCM density
    - **CP(kJ/kgK)**: Specific heat capacity
    - **Mass(kg)**: PCM mass
    - **FVR**: Fin volume ratio
    - **ES(m2)**: Extended surface   
    - **FFL(m)**: Fluid flow length
    - **HTA(m2)**: Heat transfer area   
    - **TH(°C)**: Degree of superheat
    - **TP(°C)**: Degree of subcooling
    - **LPH(L/h)**: litres per hour

    ### 💡 Usage tips：

    - Input values are rounded to six decimals
    - Automatically apply the same data scaler as in training
    - Predicted result is power density (kW/m³)
    - Hovering the mouse over a parameter name will display its description
    """)

    st.warning(f"**Current Configuration:**")
    st.warning(f"- Model URL: {MODEL_URL}")
    st.warning(f"- Scaler URL: {SCALER_URL}")

# Footer & Feedback Form
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ThermalBatteryPredictor | Project-Based Learning | Building with Streamlit"
    "</div>",
    unsafe_allow_html=True
)

# Feedback Email Configuration
DEFAULT_RECEIVER_EMAIL = "jiajiejiang@zju.edu.cn"

# Try to read receiver from secrets, fallback to default
RECEIVER_EMAIL = st.secrets.get("RECEIVER_EMAIL", DEFAULT_RECEIVER_EMAIL)

def send_feedback_email(feedback_text):
    """
    Send user feedback to the developer's email via SMTP.
    Relies on st.secrets for server credentials.
    Returns (success, message).
    """
    try:
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.zju.edu.cn")
        smtp_port = int(st.secrets.get("SMTP_PORT", 994))
        sender_email = st.secrets.get("SENDER_EMAIL", "jiajiejiang@zju.edu.cn")
        sender_password = st.secrets.get("SENDER_PASSWORD", "ZDA7yZFeFGzyQp5p")
        receiver_email = st.secrets.get("RECEIVER_EMAIL", DEFAULT_RECEIVER_EMAIL)

        if not sender_password:
            return False, "Email service not configured. Please create `.streamlit/secrets.toml` with SMTP credentials."

        # Build email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = Header("ThermalBatteryPredictor User Feedback", "utf-8")
        body = f"User feedback:\n\n{feedback_text}"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Send via SMTP (SSL)
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())

        return True, "Thank you for your feedback! It has been sent successfully."
    except Exception as e:
        return False, f"Failed to send feedback: {str(e)}"

# Feedback form placed at the bottom right corner
st.markdown("---")
col_left, col_right = st.columns([3, 1])   # Right column for feedback
with col_right:
    st.markdown(f"#### 💬 Please share your feedback with us ({RECEIVER_EMAIL})")
    with st.form(key='feedback_form'):
        feedback = st.text_area(
            "",
            placeholder="Your feedback is helpful...",
            height=100,
            label_visibility="collapsed",
            key="feedback_input"
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            if feedback.strip():
                success, msg = send_feedback_email(feedback.strip())
                if success:
                    st.success(msg)
                    # Attempt to clear the text area; ignore any widget–state conflicts
                    try:
                        st.session_state.feedback_input = ""
                        st.session_state.feedback_sent = True
                    except Exception:
                        pass
                    # Rerun to refresh the UI, clearing the text box permanently
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please enter your feedback before submitting.")