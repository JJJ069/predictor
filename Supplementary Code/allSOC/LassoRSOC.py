import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import warnings
warnings.filterwarnings("ignore")


# Data Loading
df = pd.read_excel(r'F:\PCM.xlsx', sheet_name='3600')
y = df['SOC×102(kWh/m3)']

# First correlation heatmap (all features)
columns_to_drop = ['Materials', 'PD×102(kW/m3)', 'ST', 'SC', 'FP', 'CCM',
                   'Hot fluid', 'Research type', 'Data source', 'Paper title']
df_reduced = df.drop(columns=columns_to_drop)
columns = ['SOC×102(kWh/m3)'] + [col for col in df_reduced.columns if col != 'SOC×102(kWh/m3)']
df_reduced = df_reduced[columns]
correlation_matrix = df_reduced.corr()

plt.figure(figsize=(8, 7.2))
colors = [(0, '#4C72B0'), (0.5, '#FFFFFF'), (1, '#FAA0A0')]
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

# Second correlation heatmap (after dropping more columns)
columns_to_drop = ['Materials', 'PD×102(kW/m3)', 'MT(°C)', 'TDV(m3)', 'ST', 'SC',
                   'FP', 'CCM', 'Hot fluid', 'Research type', 'Data source', 'Paper title']
df_reduced = df.drop(columns=columns_to_drop)
columns = ['SOC×102(kWh/m3)'] + [col for col in df_reduced.columns if col != 'SOC×102(kWh/m3)']
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

# Feature Matrix Preparation
X = df.drop(['PD×102(kW/m3)', 'Materials', 'SOC×102(kWh/m3)', 'MT(°C)', 'TDV(m3)',
             'ST', 'SC', 'FP', 'CCM', 'Hot fluid', 'Research type',
             'Data source', 'Paper title'], axis=1)

# Check for NaN
nan_count = X.isna().sum()
print(f"Number of NaN values in features: {nan_count}")
nan_locations = X[y.isna()]
print("Locations of NaN values in features:\n", nan_locations)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Standardization (important for Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep as DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns,
                              index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns,
                             index=X_test.index)


# Hyperparameter Optimization using GridSearchCV (Lasso)
param_grid = {
    'alpha': np.logspace(-4, 2, 20),   # Regularization strength (log scale)
    'fit_intercept': [True, False]
}

lasso = Lasso(max_iter=10000, random_state=100)  # Increase max_iter for convergence
grid_search = GridSearchCV(
    estimator=lasso,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print("Best hyperparameters:", grid_search.best_params_)
best_lasso = grid_search.best_estimator_

# Prediction Helper (non-negative)
def predict_nonneg(model, X):
    """Predict and ensure non-negative outputs."""
    pred = model.predict(X)
    return np.maximum(0, pred)

# Predictions & Evaluation
y_train_pred = predict_nonneg(best_lasso, X_train_scaled)
y_test_pred  = predict_nonneg(best_lasso, X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

results = pd.DataFrame([['Lasso Regression (GridSearchCV)',
                         train_mse, train_rmse, train_mae, train_r2,
                         test_mse, test_rmse, test_mae, test_r2]],
                       columns=['Method', 'Training MSE', 'Training RMSE',
                                'Training MAE', 'Training R2',
                                'Test MSE', 'Test RMSE',
                                'Test MAE', 'Test R2'])
print(results)

# Scatter Plots (with non-negative axis limits)
dot_size = 30

# Training
plt.figure(figsize=(6, 5))
plt.scatter(x=y_train, y=y_train_pred, c='#D9CFE8', alpha=1, s=dot_size)
z = np.polyfit(y_train, y_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
plt.xlim(left=0)
plt.ylim(bottom=0)
textstr = f'Lasso Regression: Training\nRMSE: {train_rmse:.4f}\nMAE: {train_mae:.4f}\nR²: {train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Testing
plt.figure(figsize=(6, 5))
plt.scatter(x=y_test, y=y_test_pred, c='#A3E2E8', alpha=1, s=dot_size)
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
plt.xlim(left=0)
plt.ylim(bottom=0)
textstr = f'Lasso Regression: Testing\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Combined
plt.figure(figsize=(5.35, 5))
plt.scatter(y_train, y_train_pred, alpha=1, c='#D9CFE8',
            label=f'Train (R² = {train_r2:.4f})', s=60)
plt.scatter(y_test, y_test_pred, alpha=1, c='#A3E2E8',
            label=f'Test (R² = {test_r2:.4f})', s=60)
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)
plt.xlabel(r'Actual SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.ylabel(r'Predicted SOC ×10² (kWh/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.xlim(left=0)
plt.ylim(bottom=0)
legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop, frameon=True, fancybox=True, framealpha=0.6,
           edgecolor='black', facecolor='white', loc='upper right',
           bbox_to_anchor=(0.62, 0.98))
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()


# Feature importance (coefficients) for Lasso
coefficients = best_lasso.coef_
features = X_train.columns
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(6.4, 4.6))
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'],
                color='#3aaeff', height=0.7)
plt.xlabel('Coefficient', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=10, fontname='Arial')
plt.gca().invert_yaxis()

for bar in bars:
    width = bar.get_width()
    x = bar.get_x() + width
    y = bar.get_y() + bar.get_height() / 2
    plt.text(x, y, f'{width:.4f}', ha='left', va='center', fontsize=8,
             fontname='Arial', color='black')
plt.tight_layout()
plt.show()


# 5‑Fold Cross‑Validation on best Lasso model
kf = KFold(n_splits=5, shuffle=True, random_state=100)
fold_results = []

print("\nStart 5-fold cross-validation on best Lasso model...\n")
for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
    print(f"Processing fold {fold}...")

    X_fold_train = X_train_scaled.iloc[train_index]
    X_fold_val = X_train_scaled.iloc[val_index]
    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    lasso_fold = Lasso(**grid_search.best_params_, max_iter=10000, random_state=100)
    lasso_fold.fit(X_fold_train, y_fold_train)

    y_fold_train_pred = predict_nonneg(lasso_fold, X_fold_train)
    y_fold_val_pred   = predict_nonneg(lasso_fold, X_fold_val)

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

    # Plot validation predictions
    plt.figure(figsize=(5, 5))
    plt.scatter(y_fold_val, y_fold_val_pred, c='#A3E2E8', alpha=0.8, s=30)
    min_val_fold = min(y_fold_val.min(), y_fold_val_pred.min())
    max_val_fold = max(y_fold_val.max(), y_fold_val_pred.max())
    plt.plot([min_val_fold, max_val_fold], [min_val_fold, max_val_fold], '--', c='#595959', lw=2)
    textstr_fold = (f'Fold {fold} (Validation set)\nMSE: {val_mse:.4f}\n'
                    f'RMSE: {val_rmse:.4f}\nMAE: {val_mae:.4f}\nR²: {val_r2:.4f}')
    props_fold = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.05, 0.95, textstr_fold, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=props_fold, fontname='Arial')
    plt.xlabel('Actual SOC ×10² (kWh/m³)', fontsize=14, fontname='Arial')
    plt.ylabel('Predicted SOC ×10² (kWh/m³)', fontsize=14, fontname='Arial')
    plt.title(f'Lasso Regression – Cross-validation Fold {fold}', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Summary
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
        cell_colours[-2] = ['#fffacd'] * len(col_labels)
        cell_colours[-1] = ['#fffacd'] * len(col_labels)

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
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t1.append(predicted[0])

for t2 in t_values:
    Res = pd.DataFrame([[t2, 248, 0.28, 816, 2.16, 1.484966, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t2.append(predicted[0])

for t3 in t_values:
    Res = pd.DataFrame([[t3, 187.21, 0.15, 912, 2.29, 1.659668, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t3.append(predicted[0])

for t4 in t_values:
    Res = pd.DataFrame([[t4, 155, 2.35, 300, 1.7, 0.545943, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t4.append(predicted[0])

for t5 in t_values:
    Res = pd.DataFrame([[t5, 169, 0.29, 906, 2.21, 1.648749, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t5.append(predicted[0])

for t6 in t_values:
    Res = pd.DataFrame([[t6, 227.3, 7.2, 1000, 3.22, 1.819811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t6.append(predicted[0])

for t7 in t_values:
    Res = pd.DataFrame([[t7, 186.7, 0.2, 897, 2.17, 1.632371, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t7.append(predicted[0])

for t8 in t_values:
    Res = pd.DataFrame([[t8, 174, 5.112, 1348, 2.06, 2.453106, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t8.append(predicted[0])

for t9 in t_values:
    Res = pd.DataFrame([[t9, 244.4, 0.6, 1937, 2.9, 3.524975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t9.append(predicted[0])

for t10 in t_values:
    Res = pd.DataFrame([[t10, 237.6, 0.52, 1345, 1.27, 2.447646, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t10.append(predicted[0])

for t11 in t_values:
    Res = pd.DataFrame([[t11, 169, 0.7, 1450, 1.83, 2.638727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t11.append(predicted[0])

for t12 in t_values:
    Res = pd.DataFrame([[t12, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted = predict_nonneg(best_lasso, New_Res)
    predicted_t12.append(predicted[0])

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

print(df_results.to_csv(sep='\t', index=False, float_format='%.6f'))



# SHAP interpretability analysis (using KernelExplainer for Lasso)
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Arial'

# Use a subset of training data as background (100 samples) to speed up
background = X_train_scaled.sample(n=100, random_state=100)

# Define a prediction function that returns the model's predictions (non-negative clamping not needed for SHAP)
def predict_fn(X):
    return best_lasso.predict(X)

explainer = shap.KernelExplainer(predict_fn, background)
# Compute SHAP values for a subset of the training data to reduce time
shap_values = explainer.shap_values(X_train_scaled.iloc[:100], nsamples=100)

# Summary plot (beeswarm) with custom colour map
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_scaled.iloc[:100],
                  feature_names=X_train.columns.tolist(),
                  show=False,
                  max_display=15,
                  cmap=custom_cmap)
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
shap.summary_plot(shap_values, X_train_scaled.iloc[:100],
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