import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.font_manager as font_manager
import warnings
import shap
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# Suppress warnings
warnings.filterwarnings("ignore")

# Data Loading
df = pd.read_excel(r'F:\Data S1.xlsx', sheet_name='3600')
y = df['PD×102(kW/m3)']   # Target variable

# First Correlation Heatmap (all features)
columns_to_drop = ['Materials', 'SOC×102(kWh/m3)', 'ST', 'SC', 'FP', 'CCM',
                   'Hot fluid', 'Research type', 'Data source', 'Paper title']
df_reduced = df.drop(columns=columns_to_drop)
columns = ['PD×102(kW/m3)'] + [col for col in df_reduced.columns if col != 'PD×102(kW/m3)']
df_reduced = df_reduced[columns]
correlation_matrix = df_reduced.corr()

plt.figure(figsize=(8, 7.2))
colors = [(0, '#63baf8'), (0.5, '#ffffff'), (1, '#ff7c9c')]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
ax = sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1})
ax.set_xlabel('Features', fontsize=14, fontname='Arial')
ax.set_ylabel('Features', fontsize=14, fontname='Arial')
ax.tick_params(axis='x', labelsize=9, rotation=90)
ax.tick_params(axis='y', labelsize=9, rotation=0)
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')
plt.tight_layout()
plt.show()

# Second Correlation Heatmap (after dropping more columns)
columns_to_drop = ['Materials', 'SOC×102(kWh/m3)', 'MT(°C)', 'TDV(m3)', 'ST', 'SC',
                   'FP', 'CCM', 'Hot fluid', 'Research type', 'Data source', 'Paper title']
df_reduced = df.drop(columns=columns_to_drop)
columns = ['PD×102(kW/m3)'] + [col for col in df_reduced.columns if col != 'PD×102(kW/m3)']
df_reduced = df_reduced[columns]
correlation_matrix = df_reduced.corr()

plt.figure(figsize=(8, 7.2))
ax = sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1})
ax.set_xlabel('Features', fontsize=14, fontname='Arial')
ax.set_ylabel('Features', fontsize=14, fontname='Arial')
ax.tick_params(axis='x', labelsize=9, rotation=90)
ax.tick_params(axis='y', labelsize=9, rotation=0)
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')
plt.tight_layout()
plt.show()

# Prepare Feature Matrix X
X = df.drop(['PD×102(kW/m3)', 'Materials', 'SOC×102(kWh/m3)', 'MT(°C)', 'TDV(m3)',
             'ST', 'SC', 'FP', 'CCM', 'Hot fluid', 'Research type',
             'Data source', 'Paper title'], axis=1)

# Check for NaN in features
nan_count = X.isna().sum()
print(f"Number of NaN values in features: {nan_count}")
nan_locations = X[y.isna()]
print("Locations of NaN values in features:\n", nan_locations)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Display X_test with original indices (optional)
X_test_with_indices = X_test.reset_index()
pd.set_option('display.max_rows', None)
print(X_test_with_indices)

#  Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep as DataFrames to preserve column names and indices
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns,
                              index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns,
                             index=X_test.index)

# Monotonic Constraints
feature_names = X_train.columns.tolist()

monotonic_dict = {
    'HCT': -1,    # Predicted value decreases as HCT increases
    'TC': 1,      # Predicted value increases as TC increases
    'Mass': -1,   # Predicted value decreases as Mass increases
    'FVR': 1,     # Predicted value increases as FVR increases
    'ES': 1,      # Predicted value increases as ES increases
    'HTA': 1,     # Predicted value increases as HTA increases
    'TH': 1,      # Predicted value increases as TH increases
    'LPH': 1      # Predicted value increases as LPH increases
}

monotone_constraints = []
for feat in feature_names:
    if feat in monotonic_dict:
        monotone_constraints.append(monotonic_dict[feat])
    else:
        monotone_constraints.append(0)

print("Monotone constraints (order matches features):", monotone_constraints)

# Fixed Best Hyperparameters (from prior Optuna search)
best_params = {
    'iterations': 950,
    'learning_rate': 0.09966121788824202,
    'depth': 7,
    'l2_leaf_reg': 0.25745706486652464,
    'subsample': 0.8064650007625166,
    'colsample_bylevel': 0.6414496877449262
}
print("Fixed best hyperparameters:", best_params)

# Train Final Model with Monotonic Constraints
catboost = CatBoostRegressor(
    **best_params,
    random_seed=100,
    verbose=0,
    thread_count=-1,
    monotone_constraints=monotone_constraints
)
catboost.fit(X_train_scaled, y_train)

# Predictions
y_catboost_train_pred = catboost.predict(X_train_scaled)
y_catboost_test_pred = catboost.predict(X_test_scaled)

# Evaluation Metrics
catboost_train_mse = mean_squared_error(y_train, y_catboost_train_pred)
catboost_train_rmse = np.sqrt(catboost_train_mse)
catboost_train_mae = mean_absolute_error(y_train, y_catboost_train_pred)
catboost_train_r2 = r2_score(y_train, y_catboost_train_pred)

catboost_test_mse = mean_squared_error(y_test, y_catboost_test_pred)
catboost_test_rmse = np.sqrt(catboost_test_mse)
catboost_test_mae = mean_absolute_error(y_test, y_catboost_test_pred)
catboost_test_r2 = r2_score(y_test, y_catboost_test_pred)

catboost_results = pd.DataFrame([['CatBoost',
                                   catboost_train_mse, catboost_train_rmse,
                                   catboost_train_mae, catboost_train_r2,
                                   catboost_test_mse, catboost_test_rmse,
                                   catboost_test_mae, catboost_test_r2]],
                                 columns=['Method', 'Training MSE', 'Training RMSE',
                                          'Training MAE', 'Training R2',
                                          'Test MSE', 'Test RMSE',
                                          'Test MAE', 'Test R2'])
print(catboost_results)

# Scatter Plots
dot_size = 30

# Training set
plt.figure(figsize=(6, 5))
plt.scatter(x=y_train, y=y_catboost_train_pred, c='#EC7166', alpha=1, s=dot_size)
z = np.polyfit(y_train, y_catboost_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'CatBoost: Training\nRMSE: {catboost_train_rmse:.4f}\nMAE: {catboost_train_mae:.4f}\nR²: {catboost_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Testing set
plt.figure(figsize=(6, 5))
plt.scatter(x=y_test, y=y_catboost_test_pred, c='#75C5EB', alpha=1, s=dot_size)
z = np.polyfit(y_test, y_catboost_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'CatBoost: Testing\nRMSE: {catboost_test_rmse:.4f}\nMAE: {catboost_test_mae:.4f}\nR²: {catboost_test_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Combined train/test
train_r2 = catboost_train_r2
test_r2 = catboost_test_r2
plt.figure(figsize=(5.35, 5))
plt.scatter(y_train, y_catboost_train_pred, alpha=1, c='#EC7166',
            label=f'Train (R² = {train_r2:.4f})', s=60)
plt.scatter(y_test, y_catboost_test_pred, alpha=1, c='#75C5EB',
            label=f'Test (R² = {test_r2:.4f})', s=60)
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)
plt.xlabel(r'Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.ylabel(r'Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop, frameon=True, fancybox=True, framealpha=0.6,
           edgecolor='black', facecolor='white', bbox_to_anchor=(0.62, 0.96),
           borderpad=0.4, labelspacing=0.5, handletextpad=0.2, borderaxespad=0.2)
plt.grid(False)
plt.axis('equal')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# Combined train/test
train_r2 = catboost_train_r2
test_r2 = catboost_test_r2
plt.figure(figsize=(5.35, 5))
plt.scatter(y_train, y_catboost_train_pred, alpha=1, c='#EC7166',
            label=f'Train (R² = {train_r2:.4f})', s=60)
plt.scatter(y_test, y_catboost_test_pred, alpha=1, c='#75C5EB',
            label=f'Test (R² = {test_r2:.4f})', s=60)
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)
plt.xlabel(r'Actual power density ×10² (kW/m³)', fontsize=12, fontname='Arial')
plt.ylabel(r'Predicted power density ×10² (kW/m³)', fontsize=12, fontname='Arial')
plt.xticks(fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
legend_prop = {'family': 'Arial', 'size': 12}
plt.legend(prop=legend_prop, frameon=True, fancybox=True, framealpha=0.6,
           edgecolor='black', facecolor='white', bbox_to_anchor=(0.498, 0.96),
           borderpad=0.4, labelspacing=0.5, handletextpad=0.2, borderaxespad=0.2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()


# Violin Plot of Residuals
train_residuals = y_train - y_catboost_train_pred
test_residuals = y_test - y_catboost_test_pred

residual_df = pd.DataFrame({
    'Residual': np.concatenate([train_residuals, test_residuals]),
    'Dataset': ['Train'] * len(train_residuals) + ['Test'] * len(test_residuals)
})

plt.figure(figsize=(6, 5))
sns.violinplot(x='Dataset', y='Residual', data=residual_df,
               hue='Dataset', palette={'Train': '#EC7166', 'Test': '#75C5EB'},
               inner=None, linewidth=1, legend=False)

plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=0)

plt.xlabel('Dataset', fontsize=15, fontname='Arial')
plt.ylabel('Residual (Actual - Predicted)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.yticks([-0.1, 0, 0.1])
plt.ylim(-0.1, 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Feature Importance (Bar Chart with Custom Colors)
feature_importances = catboost.feature_importances_
features = X_train.columns

# Normalize importance
feature_importances_normalized = feature_importances / np.sum(feature_importances)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_normalized
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7.6, 6.4))

font_props = font_manager.FontProperties(family='Arial', size=17, weight='bold')

colors = [
    (125/255, 198/255, 155/255, 0.7),
    (255/255, 146/255, 172/255, 1),
    (158/255, 170/255, 209/255, 0.8),
    (158/255, 170/255, 209/255, 0.8),
    (125/255, 198/255, 155/255, 0.7),
    (173/255, 97/255, 163/255, 0.6),
    (173/255, 97/255, 163/255, 0.6),
    (173/255, 97/255, 163/255, 0.6),
    (158/255, 170/255, 209/255, 0.8),
    (173/255, 97/255, 163/255, 0.6),
    (125/255, 198/255, 155/255, 0.7),
    (125/255, 198/255, 155/255, 0.7),
    (125/255, 198/255, 155/255, 0.7),
]

n_features = len(importance_df)
if n_features < 15:
    colors = colors[:n_features]
elif n_features > 15:
    import random
    base_colors = colors
    colors = []
    for i in range(n_features):
        if i < 15:
            colors.append(base_colors[i])
        else:
            r = random.randint(0, 255) / 255
            g = random.randint(0, 255) / 255
            b = random.randint(0, 255) / 255
            alpha = random.uniform(0.1, 0.9)
            colors.append((r, g, b, alpha))

for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
    plt.barh(feature, importance, height=0.7, color=colors[i])

plt.xlabel('Importance level', fontsize=17, fontname='Arial')
plt.ylabel('Feature variables', fontsize=17, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=12, fontname='Arial', color='black')
plt.gca().invert_yaxis()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Feature Importance (with Value Labels)
feature_importances = catboost.feature_importances_
feature_importances_norm = feature_importances / np.sum(feature_importances)
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_norm
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(6.4, 4.6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
                color='#3aaeff', height=0.7)
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=10, fontname='Arial')
plt.gca().invert_yaxis()

for bar in bars:
    width = bar.get_width()
    x = bar.get_x() + width
    y = bar.get_y() + bar.get_height() / 2
    plt.text(x, y, f'{width:.4f}',
             ha='left', va='center', fontsize=8,
             fontname='Arial', color='black')

plt.tight_layout()
plt.show()





# Predictions for 12 materials (t1 to t12) over a range of t values
# (These predictions will automatically respect the monotonic constraints)
# Materials and their fixed properties (except the first column which is varied)
# t1=RT35/SiC, t2=Eicosane, t3=Lauric acid, t4=Paraffin/EG, t5=Stearic acid, t6=SAT/EG,
# t7=RT60, t8=PA/Cu, t9=Ba(OH)2·8H2O, t10=Xylitol, t11=MgCl2·6H2O, t12=Erythritol

t_values = np.arange(0, 2.2, 0.05)

predicted_t1 = []
predicted_t2 = []
predicted_t3 = []
predicted_t4 = []
predicted_t5 = []
predicted_t6 = []
predicted_t7 = []
predicted_t8 = []
predicted_t9 = []
predicted_t10 = []
predicted_t11 = []
predicted_t12 = []

for t1 in t_values:
    Res = pd.DataFrame([[t1, 145, 0.82, 917, 1.9, 1.668767, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t1.append(predicted[0])

for t2 in t_values:
    Res = pd.DataFrame([[t2, 248, 0.28, 816, 2.16, 1.484966, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t2.append(predicted[0])

for t3 in t_values:
    Res = pd.DataFrame([[t3, 187.21, 0.15, 912, 2.29, 1.659668, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t3.append(predicted[0])

for t4 in t_values:
    Res = pd.DataFrame([[t4, 155, 2.35, 300, 1.7, 0.545943, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t4.append(predicted[0])

for t5 in t_values:
    Res = pd.DataFrame([[t5, 169, 0.29, 906, 2.21, 1.648749, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t5.append(predicted[0])

for t6 in t_values:
    Res = pd.DataFrame([[t6, 227.3, 7.2, 1000, 3.22, 1.819811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t6.append(predicted[0])

for t7 in t_values:
    Res = pd.DataFrame([[t7, 186.7, 0.2, 897, 2.17, 1.632371, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t7.append(predicted[0])

for t8 in t_values:
    Res = pd.DataFrame([[t8, 174, 5.112, 1348, 2.06, 2.453106, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t8.append(predicted[0])

for t9 in t_values:
    Res = pd.DataFrame([[t9, 244.4, 0.6, 1937, 2.9, 3.524975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t9.append(predicted[0])

for t10 in t_values:
    Res = pd.DataFrame([[t10, 237.6, 0.52, 1345, 1.27, 2.447646, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t10.append(predicted[0])

for t11 in t_values:
    Res = pd.DataFrame([[t11, 169, 0.7, 1450, 1.83, 2.638727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t11.append(predicted[0])

for t12 in t_values:
    Res = pd.DataFrame([[t12, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_t12.append(predicted[0])

df_results_t = pd.DataFrame({
    't': t_values,
    't1': predicted_t1,
    't2': predicted_t2,
    't3': predicted_t3,
    't4': predicted_t4,
    't5': predicted_t5,
    't6': predicted_t6,
    't7': predicted_t7,
    't8': predicted_t8,
    't9': predicted_t9,
    't10': predicted_t10,
    't11': predicted_t11,
    't12': predicted_t12
})

print(df_results_t.to_csv(sep='\t', index=False, float_format='%.6f'))

# SHAP Interpretability
plt.rcParams['font.family'] = 'Arial'

explainer = shap.TreeExplainer(catboost)
shap_values = explainer.shap_values(X_train_scaled)

# Summary plot (beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_scaled,
                  feature_names=X_train.columns.tolist(),
                  show=False,
                  max_display=15)
ax1 = plt.gca()
plt.xlabel('SHAP value (impact on model output)', fontsize=16)
for label in ax1.get_xticklabels():
    label.set_fontsize(15)
for label in ax1.get_yticklabels():
    label.set_fontsize(13)
if ax1.get_ylabel():
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=15)
plt.tight_layout()
plt.show()

# Bar plot (feature importance)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_train_scaled,
                  feature_names=X_train.columns.tolist(),
                  plot_type="bar",
                  show=False,
                  max_display=15)
ax2 = plt.gca()
plt.xlabel('Average impact on model output magnitude)', fontsize=14)
for patch in ax2.patches:
    patch.set_facecolor('#3aaeff')
    patch.set_edgecolor('#3aaeff')
for label in ax2.get_xticklabels():
    label.set_fontsize(12)
for label in ax2.get_yticklabels():
    label.set_fontsize(12)
if ax2.get_ylabel():
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=12)
plt.tight_layout()
plt.show()




# Dependence Plot: relationship between a feature and its SHAP value,
# with color indicating the interacting feature (automatically selected)
AXIS_LINEWIDTH = 1.5
top_features = importance_df['Feature'].head(1).tolist()
for feature in top_features:
    plt.figure(figsize=(7, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X_train_scaled,
        feature_names=X_train.columns.tolist(),
        show=False,
        ax=plt.gca()
    )
    plt.title(f'SHAP Dependence Plot for {feature}', fontsize=16, fontname='Arial')
    plt.xlabel(feature, fontsize=12, fontname='Arial')
    plt.ylabel('SHAP value', fontsize=12, fontname='Arial')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    plt.tight_layout()
    plt.show()


# Decision Plot: only samples with negative contributions for both HCT and Mass
feature_names = X_train.columns.tolist()
idx_HCT = feature_names.index('HCT(h)')
idx_Mass = feature_names.index('Mass(kg)')

shap_values_test = explainer.shap_values(X_test_scaled)

mask = (shap_values_test[:, idx_HCT] < 0) & (shap_values_test[:, idx_Mass] < 0)
selected_indices = np.where(mask)[0]

print(f"sample: {len(selected_indices)} / {X_test_scaled.shape[0]}")

if len(selected_indices) > 0:
    X_selected = X_test_scaled.iloc[selected_indices]
    shap_selected = shap_values_test[selected_indices, :]

    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        base_value=explainer.expected_value,
        shap_values=shap_selected,
        feature_names=feature_names,
        show=False,
        feature_order='importance'
    )
    plt.title('SHAP Decision Plot (Samples with Negative HCT & Mass Contributions)',
              fontsize=14, fontname='Arial')

    AXIS_LINEWIDTH = 1.5
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    plt.tight_layout()
    plt.show()


# Interaction Heatmap: average pairwise interaction strengths among top features
subset_size = min(100, X_train_scaled.shape[0])
X_subset = X_train_scaled.iloc[:subset_size]
shap_interaction = explainer.shap_interaction_values(X_subset)

top5_features = importance_df['Feature'].head(5).tolist()
top5_indices = [X_train.columns.tolist().index(f) for f in top5_features]

interaction_subset = np.array(shap_interaction)[:, top5_indices, :][:, :, top5_indices]
mean_interaction = np.mean(interaction_subset, axis=0)

ANNOT_FONTSIZE = 14
plt.figure(figsize=(8, 6))
sns.heatmap(
    mean_interaction,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    xticklabels=top5_features,
    yticklabels=top5_features,
    square=True,
    annot_kws={'size': ANNOT_FONTSIZE}
)
plt.title('SHAP Interaction Heatmap (Top 5 Features)', fontsize=14, fontname='Arial')
plt.tight_layout()
plt.show()



# Predictions for 3 materials
# t1=SAT/EG, t2=SAT/EG, t3=Ba(OH)2·8H2O, t3=Ba(OH)2·8H2O, t4=MgCl2·6H2O, t4=MgCl2·6H2O
# Predictions for Materials over n
n_values = np.arange(1, 8, 1)

predicted_n1 = []
predicted_n2 = []
predicted_n3 = []
predicted_n4 = []
predicted_n5 = []
predicted_n6 = []

for n1 in n_values:
    Res = pd.DataFrame([[n1, 227.3, 7.2, 1000, 3.22, 1.819811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n1.append(predicted[0])

for n2 in n_values:
    Res = pd.DataFrame([[n2, 227.3, 7.2, 1000, 3.22, 18.19811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n2.append(predicted[0])

for n3 in n_values:
    Res = pd.DataFrame([[n3, 244.4, 0.6, 1937, 2.9, 3.524975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n3.append(predicted[0])

for n4 in n_values:
    Res = pd.DataFrame([[n4, 244.4, 0.6, 1937, 2.9, 35.24975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n4.append(predicted[0])

for n5 in n_values:
    Res = pd.DataFrame([[n5, 169, 0.7, 1450, 1.83, 2.638727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n5.append(predicted[0])

for n6 in n_values:
    Res = pd.DataFrame([[n6, 169, 0.7, 1450, 1.83, 26.38727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_n6.append(predicted[0])

df_results_n = pd.DataFrame({
    'n': n_values,
    'n1': predicted_n1,
    'n2': predicted_n2,
    'n3': predicted_n3,
    'n4': predicted_n4,
    'n5': predicted_n5,
    'n6': predicted_n6
})

print(df_results_n.to_csv(sep='\t', index=False, float_format='%.6f'))

plt.figure(figsize=(10,6))
for col in df_results_n.columns[1:]:
    plt.plot(df_results_n['n'], df_results_n[col], marker='o', label=col)
plt.xlabel('Time (h)')
plt.ylabel('Predicted power density')
plt.title('Predictions for 3 materials for 2 mass')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()




# Heatmap of degree of superheat and Litres per hour
# Setup for dense grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

sigma_smooth = 8.0


x_values_dense = np.linspace(0, 50, 500)  # degree of superheat
y_values_dense = np.linspace(0, 500, 500)  # Litres per hour
X_dense, Y_dense = np.meshgrid(x_values_dense, y_values_dense)

# Batch prediction
x_flat = X_dense.ravel()
y_flat = Y_dense.ravel()

# Other values remain fixed
fixed_data = [1, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636, 0.361195, 0.4, 0.01885, 40, 10, 400]

# Density-intensive data
all_data = []
for x, y in zip(x_flat, y_flat):
    d = fixed_data.copy()
    d[10] = x   # degree of superheat
    d[12] = y  # Litres per hour
    all_data.append(d)

# Create DataFrame
all_data_df = pd.DataFrame(all_data, columns=X_train.columns)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
all_data_scaled = scaler.transform(all_data_df)

# Predicting and reshaping
Z_dense_flat = catboost.predict(all_data_scaled)
Z_dense = Z_dense_flat.reshape(Y_dense.shape)

# Application of Gauss smoothing
Z_dense = gaussian_filter(Z_dense, sigma=sigma_smooth)

# Predicted value multiplied by 100
Z_dense_display = Z_dense * 100

plt.figure(figsize=(6.3, 5))
plt.pcolormesh(X_dense, Y_dense, Z_dense_display, shading='nearest', cmap='coolwarm')

# Colorbar
cbar = plt.colorbar(pad=0.02)
cbar.set_label('Predicted power density (kW/m³)', fontname='Arial', fontsize=13)
cbar.ax.tick_params(labelsize=13)
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Self-defined ticks
plt.xticks([0, 10, 20, 30, 40, 50], fontsize=15, fontname='Arial')
plt.yticks([0, 100, 200, 300, 400, 500], fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.xlabel(r'Degree of superheat (°C)', fontname='Arial', fontsize=15)
plt.ylabel(r'Flow rate (L/h)', fontname='Arial', fontsize=15)

# Plot
plt.tight_layout()
plt.show()




# Heatmap of power density as a function of the Stefan number and the Fourier number
#   X = (TC * HCT * 3.6) / (FFL^2 * Density * CP)
#   Y = (CP * TH) / LH
# Z = predicted power density * 100

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# Indices in fixed_data (0-based) for the relevant physical properties
IDX_HCT     = 0   # HCT (h)
IDX_LH      = 1   # LH (kJ/kg)
IDX_TC      = 2   # TC (W/mK)
IDX_DENSITY = 3   # Density (kg/m³)
IDX_CP      = 4   # CP (kJ/kgK)
IDX_FFL     = 8   # FFL (m)
IDX_TH      = 10  # TH (°C)

# Directly set the range for the TRANSFORMED X and Y axes (formula values)
X_axis_min, X_axis_max = 0.0, 0.002   # Range of Fo=(TC*HCT*3.6)/(FFL^2*Density*CP)
Y_axis_min, Y_axis_max = 0.0, 0.4    # Range of Ste=(CP*TH)/LH

# Smoothing sigma
sigma_smooth = 8.0

# Fixed values (order matches the original feature list)
fixed_data = [
    5,          # HCT (h)
    333.7,      # LH (kJ/kg)
    0.8,        # TC (W/mK)
    1346,       # Density (kg/m³)
    1.98,       # CP (kJ/kgK)
    60,         # Mass (kg)
    0.025826,   # FVR
    5.4358,     # ES (m²)
    2,          # FFL (m)
    0.565487,   # HTA (m²)
    31.2,       # TH (°C)
    18.8,       # TP (°C)
    200         # LPH (L/h)
]

# Extract fixed parameters
TC      = fixed_data[IDX_TC]
Density = fixed_data[IDX_DENSITY]
CP      = fixed_data[IDX_CP]
FFL     = fixed_data[IDX_FFL]
LH      = fixed_data[IDX_LH]

# Compute the corresponding HCT and TH ranges from the desired X/Y axis ranges
# X = (TC * HCT * 3.6) / (FFL^2 * Density * CP)  =>  HCT = X * (FFL^2 * Density * CP) / (TC * 3.6)
# Y = (CP * TH) / LH  =>  TH = Y * LH / CP
denom_X = TC * 3.6
if denom_X == 0:
    raise ValueError("TC is zero, cannot compute HCT.")
HCT_min = X_axis_min * (FFL**2 * Density * CP) / denom_X
HCT_max = X_axis_max * (FFL**2 * Density * CP) / denom_X

if CP == 0:
    raise ValueError("CP is zero, cannot compute TH.")
TH_min = Y_axis_min * LH / CP
TH_max = Y_axis_max * LH / CP

print(f"Computed HCT range: [{HCT_min:.4f}, {HCT_max:.4f}]")
print(f"Computed TH range:   [{TH_min:.4f}, {TH_max:.4f}]")

# Generate dense grid over the computed HCT and TH ranges
hct_grid = np.linspace(HCT_min, HCT_max, 500)
th_grid  = np.linspace(TH_min, TH_max, 500)
X_dense, Y_dense = np.meshgrid(hct_grid, th_grid)  # X_dense = HCT, Y_dense = TH

# Flatten and prepare feature matrix
hct_flat = X_dense.ravel()
th_flat  = Y_dense.ravel()

all_data = []
for hct, th in zip(hct_flat, th_flat):
    d = fixed_data.copy()
    d[IDX_HCT] = hct
    d[IDX_TH]  = th
    all_data.append(d)

all_data_df = pd.DataFrame(all_data, columns=X_train.columns)

# Standardization and prediction
scaler = StandardScaler()
scaler.fit(X_train)
all_data_scaled = scaler.transform(all_data_df)

Z_dense_flat = catboost.predict(all_data_scaled)
Z_dense = Z_dense_flat.reshape(Y_dense.shape)

# Smoothing
Z_dense = gaussian_filter(Z_dense, sigma=sigma_smooth)
Z_dense_display = Z_dense * 100

# Compute the transformed coordinates for plotting
X_axis = (TC * X_dense * 3.6) / (FFL**2 * Density * CP)
Y_axis = (CP * Y_dense) / LH

# Plot
plt.figure(figsize=(6.3, 5))
plt.pcolormesh(X_axis, Y_axis, Z_dense_display, shading='nearest', cmap='coolwarm')

cbar = plt.colorbar(pad=0.02)
cbar.set_label('Predicted power density (kW/m³)', fontname='Arial', fontsize=13)
cbar.ax.tick_params(labelsize=13)
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.xlabel(r'$\frac{TC \cdot HCT \cdot 3.6}{FFL^2 \cdot Density \cdot CP}$',
           fontname='Arial', fontsize=15)
plt.ylabel(r'$\frac{CP \cdot TH}{LH}$',
           fontname='Arial', fontsize=15)

plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

plt.tight_layout()
plt.show()



# Predictions for experimental results
# t1=erythritol, 150oC-200Lh, t2=erythritol, 130oC-200Lh, t3=RT60, 80oC-200Lh, t4=RT60, 70oC-100Lh
e_values = np.arange(0.5, 5.5, 0.5)

predicted_e1 = []
predicted_e2 = []
predicted_e3 = []
predicted_e4 = []

for e1 in e_values:
    Res = pd.DataFrame([[e1, 333.7, 0.8, 1346, 1.98, 60, 0.025826,
                         5.4358, 2, 0.565487, 31.2, 18.8, 200]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_e1.append(predicted[0])

for e2 in e_values:
    Res = pd.DataFrame([[e2, 333.7, 0.8, 1346, 1.98, 60, 0.025826,
                         5.4358, 2, 0.565487, 11.2, 18.8, 200]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_e2.append(predicted[0])

for e3 in e_values:
    Res = pd.DataFrame([[e3, 186.7, 0.2, 897, 2.17, 36, 0.025826,
                         5.4358, 2, 0.565487, 20, 10, 200]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_e3.append(predicted[0])

for e4 in e_values:
    Res = pd.DataFrame([[e4, 186.7, 0.2, 897, 2.17, 36, 0.025826,
                         5.4358, 2, 0.565487, 10, 10, 100]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = catboost.predict(New_Res)
    predicted_e4.append(predicted[0])

df_results_e = pd.DataFrame({
    'e': e_values,
    'e1': predicted_e1,
    'e2': predicted_e2,
    'e3': predicted_e3,
    'e4': predicted_e4
})

print(df_results_e.to_csv(sep='\t', index=False, float_format='%.6f'))

plt.figure(figsize=(10,6))
for col in df_results_e.columns[1:]:
    plt.plot(df_results_e['e'], df_results_e[col], marker='o', label=col)
plt.xlabel('Time (h)')
plt.ylabel('Predicted power density')
plt.title('Predictions for 2 materials for 4 conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()



### Cumulative Probability Plot of APE
xlim_min = 0
xlim_max = 10

# Legend positions
legend1_center = (0.18, 0.93)
legend2_center = (0.68, 0.2)

# Axis line width
axis_linewidth = 1.5

# Axis label font size
label_fontsize = 18

# Tick label font size
tick_fontsize = 18

# Curve line widths
train_linewidth = 3.5
test_linewidth = 3.5

# Gray dashed line width
hline_linewidth = 2.0

# Legend font sizes
legend1_fontsize = 16
legend2_fontsize = 16

# Compute absolute percentage errors (APE) for training and test sets, excluding zero true values
def compute_ape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.array([])
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    ape = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
    return ape

ape_train = compute_ape(y_train, y_catboost_train_pred)
ape_test = compute_ape(y_test, y_catboost_test_pred)

# Empirical cumulative distribution function (ECDF)
def ecdf(data):
    if len(data) == 0:
        return np.array([]), np.array([])
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y

x_train, y_train_ecdf = ecdf(ape_train)
x_test, y_test_ecdf = ecdf(ape_test)

# Compute statistics
if len(ape_train) > 0:
    train_mean = np.mean(ape_train)
    train_median = np.median(ape_train)
    train_90 = np.percentile(ape_train, 90)
else:
    train_mean = train_median = train_90 = np.nan

if len(ape_test) > 0:
    test_mean = np.mean(ape_test)
    test_median = np.median(ape_test)
    test_90 = np.percentile(ape_test, 90)
else:
    test_mean = test_median = test_90 = np.nan

plt.figure(figsize=(7, 6))

# Plot ECDF curves with adjustable line widths
train_line, = plt.step(x_train, y_train_ecdf, where='post',
                       color='#EC7166', linewidth=train_linewidth, label='Train dataset')
test_line, = plt.step(x_test, y_test_ecdf, where='post',
                      color='#75C5EB', linewidth=test_linewidth, label='Test dataset')

# Add gray horizontal dashed lines at y=0.5 and y=0.9 (not included in legend)
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=hline_linewidth, alpha=0.7)
plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=hline_linewidth, alpha=0.7)

plt.xlim(xlim_min, xlim_max)

# Axis labels with adjustable font size
plt.xlabel('APE (%)', fontsize=label_fontsize, fontname='Arial')
plt.ylabel('Cumulative Probability', fontsize=label_fontsize, fontname='Arial')

# Tick labels with adjustable font size
plt.xticks(fontsize=tick_fontsize, fontname='Arial')
plt.yticks(fontsize=tick_fontsize, fontname='Arial')

plt.grid(False)

# Set axis spine widths
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(axis_linewidth)

# First legend (curves)
legend1 = plt.legend(
    handles=[train_line, test_line],
    labels=['Train dataset', 'Test dataset'],
    loc='center',
    bbox_to_anchor=legend1_center,
    fontsize=legend1_fontsize,
    prop={'family': 'Arial'},
    frameon=False
)
# Force apply legend font size
for text in legend1.get_texts():
    text.set_fontsize(legend1_fontsize)
plt.gca().add_artist(legend1)

# Second legend (statistics)
stat_labels = []
if not np.isnan(train_mean) and not np.isnan(test_mean):
    stat_labels.extend([
        f'Mean-Train  = {train_mean:.2f}%',
        f'Mean-Test = {test_mean:.2f}%',
        f'Median-Train = {train_median:.2f}%',
        f'Median-Test = {test_median:.2f}%',
        f'90 percentile-Train = {train_90:.2f}%',
        f'90 percentile-Test = {test_90:.2f}%'
    ])

empty_lines = [plt.Line2D([], [], color='none') for _ in stat_labels]

legend2 = plt.legend(
    handles=empty_lines,
    labels=stat_labels,
    loc='center',
    bbox_to_anchor=legend2_center,
    fontsize=legend2_fontsize,
    prop={'family': 'Arial'},
    frameon=False
)
# Force apply legend font size
for text in legend2.get_texts():
    text.set_fontsize(legend2_fontsize)
plt.gca().add_artist(legend2)

plt.tight_layout()
plt.show()



# 5‑Fold Cross‑Validation (with Monotonic Constraints)
kf = KFold(n_splits=5, shuffle=True, random_state=100)
fold_results = []
fold_predictions = []

print("Start 5-fold cross-validation...\n")

for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
    print(f"Processing fold {fold}...")

    X_fold_train = X_train_scaled.iloc[train_index]
    X_fold_val = X_train_scaled.iloc[val_index]
    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    catboost_fold = CatBoostRegressor(
        **best_params,
        random_seed=100,
        verbose=0,
        thread_count=-1,
        monotone_constraints=monotone_constraints
    )
    catboost_fold.fit(X_fold_train, y_fold_train)

    y_fold_train_pred = catboost_fold.predict(X_fold_train)
    y_fold_val_pred = catboost_fold.predict(X_fold_val)

    train_mse = mean_squared_error(y_fold_train, y_fold_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_fold_train, y_fold_train_pred)
    train_r2 = r2_score(y_fold_train, y_fold_train_pred)

    val_mse = mean_squared_error(y_fold_val, y_fold_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_fold_val, y_fold_val_pred)
    val_r2 = r2_score(y_fold_val, y_fold_val_pred)

    fold_results.append({
        'Fold': fold,
        'Train MSE': train_mse, 'Train RMSE': train_rmse,
        'Train MAE': train_mae, 'Train R2': train_r2,
        'Valid MSE': val_mse, 'Valid RMSE': val_rmse,
        'Valid MAE': val_mae, 'Valid R2': val_r2
    })
    fold_predictions.append({
        'fold': fold,
        'y_true': y_fold_val,
        'y_pred': y_fold_val_pred
    })

    # Plot validation predictions for this fold
    plt.figure(figsize=(5, 5))
    plt.scatter(y_fold_val, y_fold_val_pred, c='#75C5EB', alpha=0.8, s=30)
    min_val = min(y_fold_val.min(), y_fold_val_pred.min())
    max_val = max(y_fold_val.max(), y_fold_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)
    textstr = (f'Fold {fold} (Validation set)\nMSE: {val_mse:.4f}\n'
               f'RMSE: {val_rmse:.4f}\nMAE: {val_mae:.4f}\nR²: {val_r2:.4f}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=props, fontname='Arial')
    plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.title(f'CatBoost – Cross-validation Fold {fold}',
              fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Summary of cross‑validation results
results_df = pd.DataFrame(fold_results)
print("\n========== Detailed results of 5-fold cross-validation ==========")
print(results_df.round(4))

mean_train_r2 = results_df['Train R2'].mean()
mean_valid_r2 = results_df['Valid R2'].mean()
mean_train_rmse = results_df['Train RMSE'].mean()
mean_valid_rmse = results_df['Valid RMSE'].mean()
print(f"\nMean trained R²: {mean_train_r2:.4f} | Mean validated R²: {mean_valid_r2:.4f} "
      f"| Mean trained RMSE: {mean_train_rmse:.4f} | Mean validated RMSE: {mean_valid_rmse:.4f}")

# Table with averages and standard deviations
def draw_results_table(results_df):
    table_data = results_df[[
        'Fold',
        'Train MSE', 'Valid MSE',
        'Train MAE', 'Valid MAE',
        'Train RMSE', 'Valid RMSE',
        'Train R2', 'Valid R2'
    ]].round(4)

    avg_row = table_data[table_data.columns[1:]].mean().round(4)
    avg_row_df = pd.DataFrame([['Average'] + avg_row.tolist()],
                              columns=table_data.columns)
    std_row = table_data[table_data.columns[1:]].std(ddof=1).round(4)
    std_row_df = pd.DataFrame([['Std'] + std_row.tolist()],
                              columns=table_data.columns)

    final_table = pd.concat([table_data, avg_row_df, std_row_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    cell_text = final_table.values.tolist()
    col_labels = final_table.columns.tolist()
    col_colours = ['#f2f2f2'] * len(col_labels)
    cell_colours = [['white'] * len(col_labels) for _ in range(len(cell_text))]
    if len(cell_text) >= 2:
        cell_colours[-2] = ['#fffacd'] * len(col_labels)   # Average row
        cell_colours[-1] = ['#fffacd'] * len(col_labels)   # Std row

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colColours=col_colours, cellColours=cell_colours)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('5-Fold Cross-Validation Metrics (with Averages and Std)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

draw_results_table(results_df)

# Percentage error statistics
# Avoid division by zero， only consider samples with y_test != 0
mask = y_test != 0
if np.sum(mask) > 0:
    ape = np.abs((y_catboost_test_pred[mask] - y_test[mask]) / y_test[mask]) * 100
    mape = np.mean(ape)
    std_ape = np.std(ape)
    print("\n=== Percentage Error Statistics (Test Set) ===")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
    print(f"STD of APE (Standard Deviation of Absolute Percentage Error): {std_ape:.4f}%")
    print(f"95% Confidence interval for APE: [{mape - 1.96*std_ape:.4f}%, {mape + 1.96*std_ape:.4f}%]")
    print(f"Median APE: {np.median(ape):.4f}%")
    print(f"25th percentile: {np.percentile(ape, 25):.4f}%")
    print(f"75th percentile: {np.percentile(ape, 75):.4f}%")
else:
    mape = np.nan
    std_ape = np.nan
    print("Warning: All test targets are zero, cannot compute percentage errors.")

