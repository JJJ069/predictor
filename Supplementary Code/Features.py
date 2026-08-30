import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Global font setting
plt.rcParams['font.family'] = 'Arial'

# Data Loading
df = pd.read_excel(r'F:\Data S1.xlsx', sheet_name='3600')
y = df['PD×102(kW/m3)']  # Target variable

# Feature Matrix Preparation
# Drop non‑feature columns (target, categorical, meta‑data)
X = df.drop(['Materials', 'MT(°C)', 'ST', 'SC', 'FP', 'CCM', 'Hot fluid', 'Research type',
             'Data source', 'Paper title'], axis=1)

# Check for NaN values
nan_count = X.isna().sum()
print(f"Number of NaN values in features: {nan_count}")
nan_locations = X[y.isna()]
print("Locations of NaN values in features:\n", nan_locations)

# Plot distribution of each feature (violin + customizable boxplot)
features = X.columns.tolist()
n_features = len(features)
print(f"Number of features to plot: {n_features}")

# CUSTOM VIOLIN COLORS
violin_palette = [
    '#EB8286',          # 1
    '#7FB2D5',          # 2
    '#FFD6E7',          # 3
    '#FF86D5',          # 4
    '#F7A1C4',          # 5
    '#CDEBFA',          # 6
    '#B3E5FC',          # 7
    '#9ED8F5',          # 8
    '#81D4FA',          # 9
    '#D0D0CE',          # TDV
    '#90CAF9',          # 10
    '#B0BEC5',          # 11
    '#A7C7E7',          # 12
    '#7FB3F0',          # 13
    '#4D90FE',          # 14
    '#8DD2C5',          # 15
][:n_features]


# Boxplot style configuration (adjustable)
boxplot_config = {
    'width': 0.05,                      # Width of the box relative to the violin
    'facecolor': 'white',               # Fill color of the box (set to 'none' for transparent)
    'edgecolor': 'black',               # Border color of the box
    'edgewidth': 1,                     # Border linewidth
    'whisker_color': 'black',           # Color of whisker lines
    'whisker_width': 1,                 # Linewidth of whiskers
    'cap_color': 'black',               # Color of cap lines
    'cap_width': 1,                     # Linewidth of caps
    'median_color': 'red',              # Color of the median line
    'median_width': 2.5,                # Linewidth of the median
    'outlier_marker': 'o',              # Marker style for outliers
    'outlier_facecolor': 'gray',        # Fill color of outlier markers
    'outlier_size': 5,                  # Size of outlier markers
    'outlier_edgecolor': 'black'        # Edge color of outlier markers
}

# Loop over each feature and create a separate figure
for i, feature in enumerate(features):
    plt.figure(figsize=(7, 5))

    # Violin plot without inner boxplot
    sns.violinplot(y=df[feature], inner=None, color=violin_palette[i], linewidth=1.2)

    # Overlay boxplot with customizable parameters
    sns.boxplot(
        y=df[feature],
        width=boxplot_config['width'],
        boxprops=dict(
            facecolor=boxplot_config['facecolor'],
            edgecolor=boxplot_config['edgecolor'],
            linewidth=boxplot_config['edgewidth']
        ),
        whiskerprops=dict(
            color=boxplot_config['whisker_color'],
            linewidth=boxplot_config['whisker_width']
        ),
        capprops=dict(
            color=boxplot_config['cap_color'],
            linewidth=boxplot_config['cap_width']
        ),
        medianprops=dict(
            color=boxplot_config['median_color'],
            linewidth=boxplot_config['median_width']
        ),
        flierprops=dict(
            marker=boxplot_config['outlier_marker'],
            markerfacecolor=boxplot_config['outlier_facecolor'],
            markersize=boxplot_config['outlier_size'],
            markeredgecolor=boxplot_config['outlier_edgecolor']
        )
    )

    # Title and axis labels
    plt.title(f'Distribution of {feature}', fontsize=15, fontname='Arial')
    plt.ylabel(feature, fontsize=15, fontname='Arial')
    plt.xlabel('', fontsize=13)  # No x‑axis label

    # Force tick labels to Arial
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(15)
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(15)

    plt.tight_layout()
    plt.show()