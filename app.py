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
    page_title="Power Density Predictor",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔋 Power Density Predictor</div>', unsafe_allow_html=True)

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

# Define features
feature_config = {
    'LH(kJ/kg)': {
        'default': 333.7,
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
        'default': 1.98,
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
        'default': 0.5655,
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
    st.markdown("### 📝 Input parameters")

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
    st.markdown("### 📋 Input parameter overview")
    formatted_values = [format_value(val) for val in input_features.values()]
    features_display_df = pd.DataFrame({
        'Parameter name': list(input_features.keys()),
        'Parameter value': formatted_values,
        'Parameter description': [feature_config[name]['description'] for name in input_features.keys()]
    })

    pd.options.display.float_format = '{:.6f}'.format
    st.dataframe(features_display_df, use_container_width=True)

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
                        st.info("✅ Data preprocessing completed ( applied standardisation)")

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
                        st.info(f"**Predicting power density**: {prediction:.3f} W/cm3")

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
        st.markdown("### 📊 Parameter distribution visualisation")

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
    - **TC(W/m2K)**: Thermal conductivity
    - **CP(kJ/kgK)**: Specific heat capacity
    - **Mass(kg)**: Mass
    - **FVR**: Fin volume ratio
    - **CCM**: Close-contact melting
    - **TD(°C)**: Temperature difference
    - **TG(°C)**: Temperature gap
    - **HTA(m2)**: Heat transfer area
    - **WTC(W/m2K)**: Wall thermal conductivity
    - **FTC(W/m2K)**: Fluid thermal conductivity
    - **LPH(L/h)**: litres per hour
    - **AR**: Aspect ratio
    - **IA(°)**: Inclination angle

    ### 💡 Usage tips：

    - Input values are rounded to six decimals
    - Automatically apply the same data scaler as in training
    - Predicted result is power density (W/cm3)
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
    "Power density prediction system for phase-change thermal batteries | CatBoost regression model | Building with Streamlit"
    "</div>",
    unsafe_allow_html=True
)