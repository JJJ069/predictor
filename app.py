import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os
import plotly.graph_objects as go
import plotly.express as px

# Setting page configuration
st.set_page_config(
    page_title="ThermalBatteryDesigner",
    page_icon="🔋",
    layout="wide"
)

# Self-customizing CSS styles
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
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
    /* Custom input box style */
    .stNumberInput input {
        font-family: monospace;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    .result-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .recommendation-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔋 ThermalBatteryDesigner</div>', unsafe_allow_html=True)

# Setting GitHub repository information
GITHUB_USERNAME = "JJJ069"  # GitHub username
GITHUB_REPO = "predictor"  # GitHub repository name
GITHUB_BRANCH = "main"  # GitHub repository branch

# Model file path
MODEL_FILE_PATH = "model.pkl"  # Model filename
SCALER_FILE_PATH = "scaler.pkl"  # Standardizer filename

# Creating GitHub raw URL to download files
MODEL_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FILE_PATH}"
SCALER_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{SCALER_FILE_PATH}"

# Model management
with st.sidebar:
    st.header("🔧 Model Settings")

    # GitHub configuration display
    st.subheader("GitHub Configuration")
    github_username = st.text_input("JJJ069", value=GITHUB_USERNAME)
    github_repo = st.text_input("predictor", value=GITHUB_REPO)
    github_branch = st.text_input("main", value=GITHUB_BRANCH)

    # Update URL
    MODEL_URL = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/{github_branch}/{MODEL_FILE_PATH}"
    SCALER_URL = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/{github_branch}/{SCALER_FILE_PATH}"

    # Show current deployment
    st.info(f"Model URL: [Link]({MODEL_URL})")
    st.info(f"Scaler URL: [Link]({SCALER_URL})")

# Define PCM materials database
pcm_materials = {
    "Erythritol": {"LH": 333.7, "Density": 1346, "CP": 1.98},
    "myristic acid": {"LH": 252.92, "Density": 855.5, "CP": 2.05},
    "Beeswax": {"LH": 242.8, "Density": 970, "CP": 0.476},
    "paraffin wax-61": {"LH": 190, "Density": 850, "CP": 2.075},
    "Paraffin-27": {"LH": 231.2, "Density": 808, "CP": 2.23},
    "Paraffin-52": {"LH": 203.56, "Density": 826.24, "CP": 2.024},
    "Gallium": {"LH": 80.16, "Density": 6093, "CP": 0.3815},
    "Paraffin-55": {"LH": 102, "Density": 785, "CP": 2.85},
    "Paraffin-58": {"LH": 250, "Density": 800, "CP": 2.1},
    "Paraffin RT 82": {"LH": 176, "Density": 860, "CP": 2},
    "OP44E": {"LH": 233.5, "Density": 818, "CP": 2.15},
    "KNO3-NaNO3": {"LH": 94, "Density": 1705, "CP": 1.42},
    "Paraffin RT 25": {"LH": 170, "Density": 820, "CP": 2},
    "Paraffin wax-50": {"LH": 114.54, "Density": 820, "CP": 2},
    "paraffin wax RT64HC": {"LH": 220, "Density": 780, "CP": 2},
    "composite": {"LH": 170, "Density": 1622, "CP": 1.96},
    "Lauric acid": {"LH": 187.21, "Density": 912, "CP": 2.28},
    "NaNO3-KNO3": {"LH": 100.9, "Density": 2150, "CP": 1.27},
    "dodecanoic acid": {"LH": 182, "Density": 911, "CP": 2.2},
    "paraffin wax RT60": {"LH": 123.5, "Density": 825, "CP": 2},
    "paraffin wax-60": {"LH": 190, "Density": 850, "CP": 2.08},
    "MgCl2·6H2O": {"LH": 125, "Density": 1480, "CP": 2.32},
    "Eicosane": {"LH": 248, "Density": 816, "CP": 2.16},
    "n-eicosane": {"LH": 250, "Density": 816, "CP": 2.39},
    "Stearic acid": {"LH": 196.1, "Density": 900, "CP": 2.48},
    "NaNO3+KNO3+NaNO2": {"LH": 80, "Density": 1976.6, "CP": 1.465},
    "CH3COONa·3H2O": {"LH": 264, "Density": 1341, "CP": 2.93},
    "Molten salts": {"LH": 110, "Density": 1994, "CP": 1.626},
    "n-octadecane": {"LH": 245, "Density": 860, "CP": 2.065},
    "tetradecanol": {"LH": 227.8, "Density": 835, "CP": 2.2},
    "Tetradecane": {"LH": 226, "Density": 798, "CP": 1.95},
    "paraffin wax RT35": {"LH": 220, "Density": 805, "CP": 3.55},
    "paraffin wax RT27": {"LH": 179, "Density": 815, "CP": 2.1},
    "CaCl2.6H2O": {"LH": 170, "Density": 1622, "CP": 2.15},
    "NaNO3": {"LH": 178, "Density": 2019, "CP": 0.45},
    "D-Mannitol": {"LH": 299, "Density": 1390, "CP": 2.5},
    "paraffin wax RT82": {"LH": 176, "Density": 770, "CP": 2},
    "Ba(OH)2.8H2O": {"LH": 244.4, "Density": 1937, "CP": 2.9},
    "paraffin wax RT18": {"LH": 209, "Density": 780, "CP": 2.16},
    "paraffin wax RT58": {"LH": 230, "Density": 812, "CP": 0.9},
    "Custom": {"LH": 0, "Density": 0, "CP": 0}
}

# Heat Exchanger Selection Section
st.markdown("### 1. PCM Heat Exchanger Selection")

# Initialize session state for heat exchanger recommendation
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

with st.expander("PCM Heat Exchanger Type Selection", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        # Heat storage capacity
        Q = st.number_input(
            "Heat Storage Capacity (Q, kW·h)",
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=0.1,
            help="Total heat storage capacity of the thermal battery"
        )

        # PCM material selection
        material_options = list(pcm_materials.keys())
        selected_material = st.selectbox(
            "Select Phase Change Material (PCM)",
            options=material_options,
            index=0,
            help="Select PCM material from the database"
        )

        # Display material properties or allow custom input
        if selected_material == "Custom":
            custom_lh = st.number_input(
                "Latent Heat (kJ/kg)",
                min_value=0.0,
                max_value=5000.0,
                value=333.7,
                step=0.1
            )
            custom_density = st.number_input(
                "Density (kg/m³)",
                min_value=0.0,
                max_value=10000.0,
                value=1346.0,
                step=0.1
            )
            custom_cp = st.number_input(
                "Specific Heat Capacity (kJ/kg·K)",
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

            # Display material properties
            st.markdown('<div class="material-info">', unsafe_allow_html=True)
            st.markdown(f"**Material Properties:**")
            st.markdown(f"- Latent Heat: {LH_value} kJ/kg")
            st.markdown(f"- Density: {Density_value} kg/m³")
            st.markdown(f"- Specific Heat Capacity: {CP_value} kJ/(kg·K)")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Temperature inputs
        Ti = st.number_input(
            "Initial Temperature T$_{\mathrm{i}}$ (°C)",
            min_value=-50.0,
            max_value=500.0,
            value=20.0,
            step=0.1,
            help="Initial temperature of the thermal battery"
        )

        Th = st.number_input(
            "Maximum Temperature T$_{\mathrm{h}}$ (°C)",
            min_value=0.0,
            max_value=500.0,
            value=160.0,
            step=0.1,
            help="Maximum operating temperature"
        )

        # Heat transfer area
        HTA_hex = st.number_input(
            "Estimated Heat Transfer Area (HTA, m²)",
            min_value=0.0,
            max_value=100.0,
            value=0.5655,
            step=0.0001,
            format="%.4f",
            help="Estimated heat transfer surface area"
        )

    # Recommendation button
    if st.button("🔍 Recommended PCM Heat Exchanger", use_container_width=True):
        # Convert Q from kWh to kJ (1 kWh = 3600 kJ)
        Q_kJ = Q * 3600

        # Calculate ΔT
        delta_T = Th - Ti

        if delta_T <= 0:
            st.error("Error: Th must be greater than T$_{\mathrm{i}}$")
        else:
            # Calculate the value for heat exchanger selection
            calc_value = HTA_hex / ((Q_kJ / (CP_value * delta_T + LH_value)) / Density_value)
            st.session_state.calc_value = calc_value

            st.markdown(f"** Heat transfer surface area density (HTA/Vpcm, m²/m³) = {calc_value:.2f}")

            if calc_value >= 300:
                st.session_state.heat_exchanger_result = "Flat-Plate Heat Exchanger"
                st.session_state.comparison_needed = False
                st.session_state.show_input_prompt = True
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-text">✅ Recommended: Flat-Plate Heat Exchanger</div>', unsafe_allow_html=True)
                st.markdown("** Because heat transfer surface area density (HTA/Vpcm, m²/m³) ≥ 300")
                st.markdown('</div>', unsafe_allow_html=True)

            elif 200 < calc_value < 300:
                st.session_state.comparison_needed = True
                st.session_state.comparison_type = "two_type"
                st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
                st.markdown("### ⚖️ Further comparison: Flat-Plate vs Shell-and-Tube")
                st.markdown("**Reason:** 200 < Heat transfer surface area density (HTA/Vpcm, m²/m³) < 300")
                st.markdown("Please input the following parameters and click 'Compare Heat Exchangers'")
                st.markdown('</div>', unsafe_allow_html=True)

            else:  # calc_value <= 200
                st.session_state.comparison_needed = True
                st.session_state.comparison_type = "three_type"
                st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
                st.markdown("### ⚖️ Further comparison: Three types")
                st.markdown("**Reason:** Heat transfer surface area density (HTA/Vpcm, m²/m³) ≤ 200")
                st.markdown("Please input the following parameters and click 'Compare Heat Exchangers'")
                st.markdown('</div>', unsafe_allow_html=True)

# Show comparison inputs if needed
if st.session_state.comparison_needed:
    st.markdown("### 🔧 Parameter Comparisons")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        W = st.number_input(
            "PCM Width for Flat-Plate (W, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.03,
            step=0.001,
            format="%.3f",
            key="W_input"
        )

    with col_b:
        di = st.number_input(
            "Shell-and-Tube Inner Diameter (d$_{\mathrm{i}}$, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.018,
            step=0.001,
            format="%.3f",
            key="di_input"
        )

    with col_c:
        do = st.number_input(
            "Shell-and-Tube Outer Diameter (d$_{\mathrm{o}}$, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.03,
            step=0.001,
            format="%.3f",
            key="do_input"
        )

    if st.session_state.comparison_type == "three_type":
        col_d = st.columns(1)[0]
        D = st.number_input(
            "Spherical Capsule Outer Diameter (D, m)",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.3f",
            key="D_input"
        )

    # Compare button
    if st.button("🔍 Compare Heat Exchangers", key="compare_button"):
        if st.session_state.comparison_type == "two_type":
            # Comparison for two types
            comparison_value = 4 * W * di / (do ** 2 - di ** 2)
            st.markdown(f"**Ratio of heat transfer surface area density: $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ = {comparison_value:.3f}")

            if comparison_value >= 1:
                st.session_state.heat_exchanger_result = "Shell-and-Tube Heat Exchanger"
                st.session_state.show_input_prompt = True
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-text">✅ Recommended: Shell-and-Tube Heat Exchanger</div>', unsafe_allow_html=True)
                st.markdown("**Because the ratio of heat transfer surface area density:** $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ ≥ 1")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.session_state.heat_exchanger_result = "Flat-Plate Heat Exchanger"
                st.session_state.show_input_prompt = True
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-text">✅ Recommended: Flat-Plate Heat Exchanger</div>', unsafe_allow_html=True)
                st.markdown("**Because the ratio of heat transfer surface area density:** $\\frac{{4Wd_i}}{{d_o^2 - d_i^2}}$ < 1")
                st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.comparison_type == "three_type":
            # Comparison for three types
            comparison_value1 = 2 * D * di / (3 * (do ** 2 - di ** 2))
            st.markdown(f"**Ratio of heat transfer surface area density: $\\frac{{2Dd_i}}{{3(d_o^2 - d_i^2)}}$ = {comparison_value1:.3f}")

            if comparison_value1 >= 1:
                st.session_state.heat_exchanger_result = "Shell-and-Tube Heat Exchanger"
                st.session_state.show_input_prompt = True
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-text">✅ Recommended: Shell-and-Tube Heat Exchanger</div>', unsafe_allow_html=True)
                st.markdown("**Because the ratio of heat transfer surface area density:** $\\frac{{2Dd_i}}{{3(d_o^2 - d_i^2)}}$ ≥ 1")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                comparison_value2 = D / (6 * W)
                st.markdown(f"**D/6W:** {comparison_value2:.3f}")

                if comparison_value2 > 1:
                    st.session_state.heat_exchanger_result = "Flat-Plate Heat Exchanger"
                    st.session_state.show_input_prompt = True
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown('<div class="recommendation-text">✅ Recommended: Flat-Plate Heat Exchanger</div>', unsafe_allow_html=True)
                    st.markdown("**Because the ratio of heat transfer surface area density:** D/6W > 1")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.session_state.heat_exchanger_result = "Spherical Capsule Heat Exchanger"
                    st.session_state.show_input_prompt = True
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown('<div class="recommendation-text">✅ Recommended: Spherical Capsule Heat Exchanger</div>', unsafe_allow_html=True)
                    st.markdown("**Because the ratio of heat transfer surface area density:** D/6W ≤ 1")
                    st.markdown('</div>', unsafe_allow_html=True)

# Show final recommendation if available
if st.session_state.heat_exchanger_result and not st.session_state.comparison_needed:
    st.markdown(f"**Final Recommendation:** {st.session_state.heat_exchanger_result}")

# Show instruction after heat exchanger selection only when heat exchanger is selected
if st.session_state.show_input_prompt:
    st.info(
        "Please input the design parameters below after completing heat exchanger selection, then click 'Start' for power density prediction.")

# Define features configuration (with updated default values from heat exchanger section)
feature_config = {
    'LH(kJ/kg)': {
        'default': LH_value if 'LH_value' in locals() else 333.7,
        'min': 0,
        'max': 5000,
        'step': 0.0001,
        'description': 'Latent heat'
    },
    'MT(°C)': {
        'default': 118.8,
        'min': 0,
        'max': 500,
        'step': 0.0001,
        'description': 'Melting temperature'
    },
    'TC(W/mK)': {
        'default': 0.8,
        'min': 0.0,
        'max': 10.0,
        'step': 0.0001,
        'description': 'Thermal conductivity'
    },
    'CP(kJ/kgK)': {
        'default': CP_value if 'CP_value' in locals() else 1.98,
        'min': 0.0,
        'max': 10.0,
        'step': 0.0001,
        'description': 'Specific heat capacity'
    },
    'Mass(kg)': {
        'default': 60,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.0001,
        'description': 'Mass'
    },
    'FVR': {
        'default': 0.0245,
        'min': 0.0,
        'max': 1.0,
        'step': 0.0001,
        'description': 'Fin volume ratio'
    },
    'CCM': {
        'default': 0,
        'min': 0,
        'max': 1,
        'step': 1,
        'description': 'Close-contact melting'
    },
    'TD(°C)': {
        'default': 41.2,
        'min': 0.0,
        'max': 200.0,
        'step': 0.0001,
        'description': 'Temperature difference'
    },
    'TG(°C)': {
        'default': 98.8,
        'min': 0.0,
        'max': 200.0,
        'step': 0.0001,
        'description': 'Temperature gap'
    },
    'HTA(m2)': {
        'default': HTA_hex if 'HTA_hex' in locals() else 0.5655,
        'min': 0.0,
        'max': 100.0,
        'step': 0.0001,
        'description': 'Heat transfer area'
    },
    'WTC(W/mK)': {
        'default': 400,
        'min': 0.0,
        'max': 2000.0,
        'step': 0.0001,
        'description': 'Wall thermal conductivity'
    },
    'FTC(W/mK)': {
        'default': 0.125,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.0001,
        'description': 'Fluid thermal conductivity'
    },
    'LPH(L/h)': {
        'default': 300,
        'min': 0.0,
        'max': 1000.0,
        'step': 0.0001,
        'description': 'litres per hour'
    },
    'AR': {
        'default': 0.0815,
        'min': 0.01,
        'max': 100.0,
        'step': 0.0001,
        'description': 'Aspect ratio'
    },
    'IA(°)': {
        'default': 0,
        'min': 0.0,
        'max': 90.0,
        'step': 0.0001,
        'description': 'Inclined angle'
    }
}


@st.cache_resource
def download_from_github(url, file_type="model"):
    """Download files from GitHub"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request is successful

        # Convert content to byte streams
        file_content = io.BytesIO(response.content)

        # Loading by file type
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


# Load the model and standardizer from GitHub
@st.cache_resource
def load_resources():
    """Load the model and standardizer"""
    model = download_from_github(MODEL_URL, "model")
    scaler = download_from_github(SCALER_URL, "scaler")
    return model, scaler


model, scaler = load_resources()


def format_value(value):
    return float(f"{value:.6f}")


# Data preprocessing
def preprocess_input(input_features, scaler):
    """Apply the same preprocessing to the input data as used for training"""
    try:
        # Create the DataFrame, ensuring the feature order matches those used for training
        feature_order = [
            'LH(kJ/kg)', 'MT(°C)', 'TC(W/mK)', 'CP(kJ/kgK)', 'Mass(kg)',
            'FVR', 'CCM', 'TD(°C)', 'TG(°C)', 'HTA(m2)',
            'WTC(W/mK)', 'FTC(W/mK)', 'LPH(L/h)', 'AR', 'IA(°)'
        ]

        # Ensure input features are correctly ordered
        ordered_features = {feature: input_features[feature] for feature in feature_order}
        input_df = pd.DataFrame([ordered_features])

        # Applying standardization (same as for training)
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_df_scaled = pd.DataFrame(input_scaled, columns=feature_order)
            return input_df_scaled
        else:
            return input_df

    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        return None


# Main content region
if model is not None and scaler is not None:
    # Displaying successful loading information
    st.sidebar.success("✅ All resources loaded successfully!")

    # Feature input section
    st.markdown("")  #
    st.markdown("")  #
    st.markdown("")  #
    st.markdown("### 2. Input Parameters to Predict Power Density")

    # Create 5 columns, each containing 3 features
    col1, col2, col3, col4, col5 = st.columns(5)
    input_features = {}

    # First column
    with col1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        for feature in list(feature_config.keys())[:3]:
            config = feature_config[feature]
            input_value = st.number_input(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=float(config['step']),
                help=config['description'],
                key=f"feature_{feature}",
                format="%.4f"
            )
            input_features[feature] = format_value(input_value)
        st.markdown('</div>', unsafe_allow_html=True)

    # Second column
    with col2:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        for feature in list(feature_config.keys())[3:6]:
            config = feature_config[feature]
            input_value = st.number_input(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=float(config['step']),
                help=config['description'],
                key=f"feature_{feature}",
                format="%.4f"
            )
            input_features[feature] = format_value(input_value)
        st.markdown('</div>', unsafe_allow_html=True)

    # Third column
    with col3:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        for feature in list(feature_config.keys())[6:9]:
            config = feature_config[feature]
            input_value = st.number_input(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=float(config['step']),
                help=config['description'],
                key=f"feature_{feature}",
                format="%.4f"
            )
            input_features[feature] = format_value(input_value)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fourth column
    with col4:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        for feature in list(feature_config.keys())[9:12]:
            config = feature_config[feature]
            input_value = st.number_input(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=float(config['step']),
                help=config['description'],
                key=f"feature_{feature}",
                format="%.4f"
            )
            input_features[feature] = format_value(input_value)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fifth column
    with col5:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        for feature in list(feature_config.keys())[12:]:
            config = feature_config[feature]
            input_value = st.number_input(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=float(config['step']),
                help=config['description'],
                key=f"feature_{feature}",
                format="%.4f"
            )
            input_features[feature] = format_value(input_value)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display input feature table
    st.markdown("### 📋 Input Parameter Overview")
    formatted_values = [format_value(val) for val in input_features.values()]
    param_index = list(range(1, len(input_features) + 1))
    features_display_df = pd.DataFrame({
        'No.': param_index,
        'Parameter name': list(input_features.keys()),
        'Parameter value': formatted_values,
        'Parameter description': [feature_config[name]['description'] for name in input_features.keys()]
    })
    pd.options.display.float_format = '{:.4f}'.format
    st.dataframe(features_display_df, use_container_width=True, hide_index=True)


    # Predicting button and result display
    st.markdown("---")

    col_pred_left, col_pred_right = st.columns([1, 1])

    with col_pred_left:
        if st.button("🚀 Start", use_container_width=True):
            with st.spinner("Calculating the predicted value..."):
                try:
                    # Data preprocessing (standardization)
                    processed_data = preprocess_input(input_features, scaler)

                    if processed_data is not None:
                        st.info("✅ Data preprocessing completed ( applied standardization)")

                        # Make predictions
                        prediction = model.predict(processed_data)[0]

                        # Display prediction results
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>📈 Predicted value</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:.3f} W/cm<sup>3</sup></h1>
                            <p>According to {len(input_features)} thermodynamic parameters to calculate as</p>
                            <p>Power density predictive value</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display detailed predictive information
                        st.info(f"**Predicted power density**: {prediction:.3f} W/cm³")

                        # Display the distribution chart of input parameters
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(input_features.keys()),
                                y=list(input_features.values()),
                                marker_color='lightblue',
                                name='Input parameter values'
                            )
                        ])
                        fig.update_layout(
                            title="Input parameter distribution",
                            xaxis_title="Parameter name",
                            yaxis_title="Parameter value",
                            showlegend=True,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Errors occurred during the predicting process: {str(e)}")

    with col_pred_right:
        st.markdown("### 📊 Parameter distribution visualization")

        # Create a pie chart of parameter distributions
        feature_values = list(input_features.values())
        feature_names = list(input_features.keys())

        normalized_values = [abs(v) / max(abs(v) for v in feature_values) for v in feature_values]

        fig_pie = px.pie(
            values=normalized_values,
            names=feature_names,
            title="Parameter value relative distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Display model information
        with st.expander("🔍 Model Information"):
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.write("Model parameters:")
                    st.json(params)
            except:
                st.write("Unable to capture model parameter details")

else:
    # Failed loading page
    st.markdown("""
    ## ⚠️ Unable to load model

    Please check the following:

    1. **GitHub Configuration** - Ensure the GitHub username, repository name, and branch are correct in the sidebar
    2. **File Names** - Make sure the model files are named correctly:
       - `model.pkl` for the model
       - `scaler.pkl` for the scaler
    3. **File Locations** - Ensure files are in the root directory of your GitHub repository
    4. **File Accessibility** - Check that the repository is public or you have access if it's private

    ### 🔧 Configuration Instructions:

    Update the following in the sidebar:
    - **GitHub Username**: Your GitHub username
    - **Repository Name**: The name of your repository containing the model files
    - **Branch**: Usually "main" or "master"

    ### 📋 System parameters description：

    - **LH(kJ/kg)**: Latent heat
    - **MT(°C)**: Melt temperature
    - **TC(W/mK)**: Thermal conductivity
    - **CP(kJ/kgK)**: Specific heat capacity
    - **Mass(kg)**: Mass
    - **FVR**: Fin volume ratio
    - **CCM**: Close-contact melting
    - **TD(°C)**: Temperature difference
    - **TG(°C)**: Temperature gap
    - **HTA(m²)**: Heat transfer area
    - **WTC(W/mK)**: Wall thermal conductivity
    - **FTC(W/mK)**: Fluid thermal conductivity
    - **LPH(L/h)**: litres per hour
    - **AR**: Aspect ratio
    - **IA(°)**: Inclination angle

    ### 💡 Usage tips：

    - Input values are rounded to six decimals
    - Automatically apply the same data scaler as in training
    - Predicted result is power density (W/cm³)
    - Hovering the mouse over a parameter name will display its description
    """)

    # Display current configuration
    st.warning(f"**Current Configuration:**")
    st.warning(f"- Model URL: {MODEL_URL}")
    st.warning(f"- Scaler URL: {SCALER_URL}")

# Footer Information
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ThermalBatteryDesigner | Machine learning model | Building with Streamlit"
    "</div>",
    unsafe_allow_html=True
)