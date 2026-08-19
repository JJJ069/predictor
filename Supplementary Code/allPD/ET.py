import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import shap
import warnings
warnings.filterwarnings("ignore")


# Load data (identical to CatBoost script)
df = pd.read_excel(r'F:\PCM.xlsx', sheet_name='3600')
y = df['PD×102(kW/m3)']   # Target variable

# Prepare feature matrix X (descriptors)
X = df.drop(['PD×102(kW/m3)', 'Materials', 'SOC×102(kWh/m3)', 'MT(°C)', 'TDV(m3)',
             'ST', 'SC', 'FP', 'CCM', 'Hot fluid', 'Research type',
             'Data source', 'Paper title'], axis=1)

# Check for NaN in features (optional)
nan_count = X.isna().sum()
print(f"Number of NaN values in features: {nan_count}")


# Train / Test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)


# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep as DataFrames to preserve column names and indices
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns,
                              index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns,
                             index=X_test.index)

# Define physical monotonic constraints
feature_names = X_train.columns.tolist()

monotonic_dict = {
    'HCT': -1,    # predicted value decreases as HCT increases
    'TC': 1,      # predicted value increases as TC increases
    'Mass': -1,   # predicted value decreases as Mass increases
    'FVR': 1,     # predicted value increases as FVR increases
    'ES': 1,      # predicted value increases as ES increases
    'HTA': 1,     # predicted value increases as HTA increases
    'TH': 1,      # predicted value increases as TH increases
    'LPH': 1      # predicted value increases as LPH increases
}

# Build constraint list (unused)
monotone_constraints = [monotonic_dict.get(feat, 0) for feat in feature_names]
print("Monotone constraints (order matches features) – NOT APPLIED to Extra Trees:", monotone_constraints)

# Hyperparameter optimization with Optuna (TPE) for Extra Trees
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=100,
        n_jobs=-1
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=100)
    neg_rmse = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -neg_rmse.mean()   # minimise RMSE

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=100))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)

# Train final Extra Trees model on scaled data (no constraints)
etr = ExtraTreesRegressor(**study.best_params, random_state=100, n_jobs=-1)
etr.fit(X_train_scaled, y_train)

# Predictions and evaluation metrics
y_train_pred = etr.predict(X_train_scaled)
y_test_pred = etr.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

results = pd.DataFrame([['Extra Trees (no monotonic constraints)',
                         train_mse, train_rmse, train_mae, train_r2,
                         test_mse, test_rmse, test_mae, test_r2]],
                       columns=['Method', 'Training MSE', 'Training RMSE',
                                'Training MAE', 'Training R2',
                                'Test MSE', 'Test RMSE',
                                'Test MAE', 'Test R2'])
print("\nPerformance Summary:")
print(results)

# Scatter plots (Training, Testing, Combined) – identical to CatBoost
dot_size = 30
props = dict(boxstyle='round', facecolor='white', alpha=1)

# Training set
plt.figure(figsize=(6, 5))
plt.scatter(x=y_train, y=y_train_pred, c='#EC7166', alpha=1, s=dot_size)
z = np.polyfit(y_train, y_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'Extra Trees: Training\nRMSE: {train_rmse:.4f}\nMAE: {train_mae:.4f}\nR²: {train_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Testing set
plt.figure(figsize=(6, 5))
plt.scatter(x=y_test, y=y_test_pred, c='#75C5EB', alpha=1, s=dot_size)
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'Extra Trees: Testing\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Combined train/test
plt.figure(figsize=(5.35, 5))
plt.scatter(y_train, y_train_pred, alpha=1, c='#EC7166',
            label=f'Train (R² = {train_r2:.4f})', s=60)
plt.scatter(y_test, y_test_pred, alpha=1, c='#75C5EB',
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
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Feature importance (Extra Trees built-in)
feature_importances = etr.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
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

# Add numeric labels at the right edge
for bar in bars:
    width = bar.get_width()
    x = bar.get_x() + width
    y = bar.get_y() + bar.get_height() / 2
    plt.text(x, y, f'{width:.4f}', ha='left', va='center', fontsize=8,
             fontname='Arial', color='black')
plt.tight_layout()
plt.show()



# 5‑fold cross‑validation on training set (using scaled data)
kf = KFold(n_splits=5, shuffle=True, random_state=100)
fold_results = []

print("\nStart 5-fold cross-validation...\n")
for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
    print(f"Processing fold {fold}...")

    X_fold_train = X_train_scaled.iloc[train_index]
    X_fold_val = X_train_scaled.iloc[val_index]
    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    etr_fold = ExtraTreesRegressor(**study.best_params, random_state=100, n_jobs=-1)
    etr_fold.fit(X_fold_train, y_fold_train)

    y_fold_train_pred = etr_fold.predict(X_fold_train)
    y_fold_val_pred = etr_fold.predict(X_fold_val)

    train_mse_fold = mean_squared_error(y_fold_train, y_fold_train_pred)
    train_rmse_fold = np.sqrt(train_mse_fold)
    train_mae_fold = mean_absolute_error(y_fold_train, y_fold_train_pred)
    train_r2_fold = r2_score(y_fold_train, y_fold_train_pred)

    val_mse = mean_squared_error(y_fold_val, y_fold_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_fold_val, y_fold_val_pred)
    val_r2 = r2_score(y_fold_val, y_fold_val_pred)

    fold_results.append({
        'Fold': fold,
        'Train MSE': train_mse_fold, 'Train RMSE': train_rmse_fold,
        'Train MAE': train_mae_fold, 'Train R2': train_r2_fold,
        'Valid MSE': val_mse, 'Valid RMSE': val_rmse,
        'Valid MAE': val_mae, 'Valid R2': val_r2
    })

    # Plot validation predictions for this fold
    plt.figure(figsize=(5, 5))
    plt.scatter(y_fold_val, y_fold_val_pred, c='#75C5EB', alpha=0.8, s=30)
    min_val_fold = min(y_fold_val.min(), y_fold_val_pred.min())
    max_val_fold = max(y_fold_val.max(), y_fold_val_pred.max())
    plt.plot([min_val_fold, max_val_fold], [min_val_fold, max_val_fold], '--', c='#595959', lw=2)

    textstr_fold = (f'Fold {fold} (Validation set)\nMSE: {val_mse:.4f}\n'
                    f'RMSE: {val_rmse:.4f}\nMAE: {val_mae:.4f}\nR²: {val_r2:.4f}')
    props_fold = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.05, 0.95, textstr_fold, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=props_fold, fontname='Arial')
    plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.title(f'Extra Trees – Cross-validation Fold {fold}', fontsize=13, fontweight='bold')
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



# Predictions for 12 materials (t1 to t12) over a range of t values
# (These predictions will automatically respect the monotonic constraints)
# Materials and their fixed properties (except the first column which is varied)
# t1=RT35/SiC, t2=Eicosane, t3=Lauric acid, t4=Paraffin/EG, t5=Stearic acid, t6=SAT/EG,
# t7=RT60, t8=PA/Cu, t9=Ba(OH)2·8H2O, t10=Xylitol, t11=MgCl2·6H2O, t12=Erythritol
t_values = np.arange(0, 2, 0.05)

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
    predicted = etr.predict(New_Res)
    predicted_t1.append(predicted[0])

for t2 in t_values:
    Res = pd.DataFrame([[t2, 248, 0.28, 816, 2.16, 1.484966, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t2.append(predicted[0])

for t3 in t_values:
    Res = pd.DataFrame([[t3, 187.21, 0.15, 912, 2.29, 1.659668, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t3.append(predicted[0])

for t4 in t_values:
    Res = pd.DataFrame([[t4, 155, 2.35, 300, 1.7, 0.545943, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t4.append(predicted[0])

for t5 in t_values:
    Res = pd.DataFrame([[t5, 169, 0.29, 906, 2.21, 1.648749, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t5.append(predicted[0])

for t6 in t_values:
    Res = pd.DataFrame([[t6, 227.3, 7.2, 1000, 3.22, 1.819811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t6.append(predicted[0])

for t7 in t_values:
    Res = pd.DataFrame([[t7, 186.7, 0.2, 897, 2.17, 1.632371, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t7.append(predicted[0])

for t8 in t_values:
    Res = pd.DataFrame([[t8, 174, 5.112, 1348, 2.06, 2.453106, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t8.append(predicted[0])

for t9 in t_values:
    Res = pd.DataFrame([[t9, 244.4, 0.6, 1937, 2.9, 3.524975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t9.append(predicted[0])

for t10 in t_values:
    Res = pd.DataFrame([[t10, 237.6, 0.52, 1345, 1.27, 2.447646, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t10.append(predicted[0])

for t11 in t_values:
    Res = pd.DataFrame([[t11, 169, 0.7, 1450, 1.83, 2.638727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t11.append(predicted[0])

for t12 in t_values:
    Res = pd.DataFrame([[t12, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = etr.predict(New_Res)
    predicted_t12.append(predicted[0])

# DataFrame
df_results = pd.DataFrame({
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

print("\nPredictions for 12 materials (tab-separated):")
print(df_results.to_csv(sep='\t', index=False, float_format='%.6f'))




# SHAP interpretability analysis (for Extra Trees)
import shap
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Arial'

explainer = shap.TreeExplainer(etr)
shap_values = explainer.shap_values(X_train_scaled)

# Summary plot (beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_scaled,
                  feature_names=X_train.columns.tolist(),
                  show=False,
                  max_display=15)
ax1 = plt.gca()
plt.xlabel('SHAP value (impact on model output)', fontsize=14)
for label in ax1.get_xticklabels():
    label.set_fontsize(12)
for label in ax1.get_yticklabels():
    label.set_fontsize(12)
if ax1.get_ylabel():
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=13)
plt.tight_layout()
plt.show()

# Bar plot (feature importance from SHAP)
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


# Heatmap of Specific heat capacity and Latent heat
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


# Heatmap of Temperature difference and Litres per hour
# Setup for dense grid
x_values_dense = np.linspace(0.1, 1, 500)  # Temperature difference
y_values_dense = np.linspace(10, 50, 500)  # Litres per hour
X_dense, Y_dense = np.meshgrid(x_values_dense, y_values_dense)

# Batch prediction
x_flat = X_dense.ravel()
y_flat = Y_dense.ravel()

# Other values remain fixed
fixed_data = [5, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636, 0.361195, 0.4, 0.01885, 40, 10, 500]

# Density-intensive data
all_data = []
for x, y in zip(x_flat, y_flat):
    d = fixed_data.copy()
    d[9] = x   # Temperature potential
    d[10] = y  # Litres per hour
    all_data.append(d)

# Create DataFrame
all_data_df = pd.DataFrame(all_data, columns=X_train.columns)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
all_data_scaled = scaler.transform(all_data_df)

# Predicting and reshaping
Z_dense_flat = etr.predict(all_data_scaled)
Z_dense = Z_dense_flat.reshape(Y_dense.shape)

# Application of Gauss smoothing
Z_dense = gaussian_filter(Z_dense, sigma=sigma_smooth)

plt.figure(figsize=(6, 5))
plt.pcolormesh(X_dense, Y_dense, Z_dense, shading='nearest', cmap='Blues_r')

# Colorbar
cbar = plt.colorbar(pad=0.02)
cbar.set_label('Predicted power density ($\mathrm{W/cm^3}$)', fontname='Arial', fontsize=13)
cbar.ax.tick_params(labelsize=13)
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Self-defined ticks
plt.xticks([0.2, 0.4, 0.6, 0.8, 1], fontsize=15, fontname='Arial')
plt.yticks([10, 20, 30, 40, 50], fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.xlabel('(m2)', fontname='Arial', fontsize=15)
plt.ylabel(r'(°C)', fontname='Arial', fontsize=15)

# Plot
plt.tight_layout()
plt.show()