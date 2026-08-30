import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# Load data
df = pd.read_excel(r'F:\Data S1.xlsx', sheet_name='3600')
y = df['PD×102(kW/m3)']   # Target variable

# Prepare feature matrix X
X = df.drop(['PD×102(kW/m3)', 'Materials', 'SOC×102(kWh/m3)', 'MT(°C)', 'TDV(m3)',
             'ST', 'SC', 'FP', 'CCM', 'Hot fluid', 'Research type',
             'Data source', 'Paper title'], axis=1)

# Check for NaN in features
nan_count = X.isna().sum()
print(f"Number of NaN values in features: {nan_count}")
nan_locations = X[y.isna()]
print("Locations of NaN values in features:\n", nan_locations)

# Train / Test split (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

# Display X_test with original indices (optional)
X_test_with_indices = X_test.reset_index()
pd.set_option('display.max_rows', None)
print(X_test_with_indices)


# Define monotonic constraints (for documentation only)
# NOTE: Ridge does NOT support monotonic constraints.

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

monotone_constraints = [monotonic_dict.get(feat, 0) for feat in feature_names]
print("Monotone constraints (order matches features) – NOT APPLIED to Ridge:", monotone_constraints)

# GridSearchCV for Ridge (with scaling pipeline)
pipeline_ridge = make_pipeline(
    StandardScaler(),
    Ridge(random_state=100)
)

param_grid_ridge = {
    'ridge__alpha': [1, 10, 50, 100, 200],
    'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}

grid_search_ridge = GridSearchCV(
    estimator=pipeline_ridge,
    param_grid=param_grid_ridge,
    scoring='neg_root_mean_squared_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=100),
    n_jobs=-1,
    verbose=1
)

print("\nStarting GridSearchCV for Ridge Regression...")
grid_search_ridge.fit(X_train, y_train)

print("Best hyperparameters:", grid_search_ridge.best_params_)
print("Best cross-validation RMSE:", -grid_search_ridge.best_score_)

# Best estimator
best_ridge = grid_search_ridge.best_estimator_

# Predictions and evaluation metrics
y_train_pred = best_ridge.predict(X_train)
y_test_pred = best_ridge.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

results = pd.DataFrame([['Ridge (no monotonic constraints)',
                         train_mse, train_rmse, train_mae, train_r2,
                         test_mse, test_rmse, test_mae, test_r2]],
                       columns=['Method', 'Training MSE', 'Training RMSE',
                                'Training MAE', 'Training R2',
                                'Test MSE', 'Test RMSE',
                                'Test MAE', 'Test R2'])
print("\nPerformance Summary:")
print(results)

# Scatter plots – filter out negative predictions for plotting only
# Metrics remain computed on all data.
dot_size = 30
props = dict(boxstyle='round', facecolor='white', alpha=1)

# Training set – keep only points with non‑negative predictions
mask_train = y_train_pred >= 0
plt.figure(figsize=(6, 5))
plt.scatter(x=y_train[mask_train], y=y_train_pred[mask_train],
            c='#EC7166', alpha=1, s=dot_size)
if np.sum(mask_train) > 1:
    z = np.polyfit(y_train[mask_train], y_train_pred[mask_train], 1)
    p = np.poly1d(z)
    plt.plot(y_train[mask_train], p(y_train[mask_train]), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'Ridge: Training (negative preds omitted)\nRMSE: {train_rmse:.4f}\nMAE: {train_mae:.4f}\nR²: {train_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Testing set – keep only non‑negative predictions
mask_test = y_test_pred >= 0
plt.figure(figsize=(6, 5))
plt.scatter(x=y_test[mask_test], y=y_test_pred[mask_test],
            c='#75C5EB', alpha=1, s=dot_size)
if np.sum(mask_test) > 1:
    z = np.polyfit(y_test[mask_test], y_test_pred[mask_test], 1)
    p = np.poly1d(z)
    plt.plot(y_test[mask_test], p(y_test[mask_test]), c='#595959', linestyle=':', linewidth=2)
plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)
textstr = f'Ridge: Testing (negative preds omitted)\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()

# Combined train/test – filter each set separately
mask_train_comb = y_train_pred >= 0
mask_test_comb = y_test_pred >= 0
plt.figure(figsize=(5.35, 5))
plt.scatter(y_train[mask_train_comb], y_train_pred[mask_train_comb],
            alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.4f})', s=60)
plt.scatter(y_test[mask_test_comb], y_test_pred[mask_test_comb],
            alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.4f})', s=60)

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

# Feature importance: Ridge has coefficients (absolute values as importance)
# Extract the scaler and the ridge model from the pipeline
scaler = best_ridge.named_steps['standardscaler']
ridge_model = best_ridge.named_steps['ridge']
coefs = ridge_model.coef_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs,
    'Abs_Coefficient': np.abs(coefs)
}).sort_values(by='Abs_Coefficient', ascending=False)

plt.figure(figsize=(6.4, 4.6))
bars = plt.barh(importance_df['Feature'], importance_df['Abs_Coefficient'],
                color='#3aaeff', height=0.7)
plt.xlabel('Absolute Coefficient (importance)', fontsize=15, fontname='Arial')
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

# 5‑fold cross‑validation on training set (with best pipeline)
kf = KFold(n_splits=5, shuffle=True, random_state=100)
fold_results = []

# Extract best hyperparameters
best_alpha = grid_search_ridge.best_params_['ridge__alpha']
best_solver = grid_search_ridge.best_params_.get('ridge__solver', 'auto')

print("\nStart 5-fold cross-validation (with best Ridge parameters)...\n")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"Processing fold {fold}...")

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Build pipeline with the best parameters
    ridge_fold = make_pipeline(
        StandardScaler(),
        Ridge(alpha=best_alpha, solver=best_solver, random_state=100)
    )
    ridge_fold.fit(X_fold_train, y_fold_train)

    y_fold_train_pred = ridge_fold.predict(X_fold_train)
    y_fold_val_pred = ridge_fold.predict(X_fold_val)

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

    # Plot validation predictions for this fold (filter negative predictions)
    mask_val = y_fold_val_pred >= 0
    plt.figure(figsize=(5, 5))
    plt.scatter(y_fold_val[mask_val], y_fold_val_pred[mask_val],
                c='#75C5EB', alpha=0.8, s=30)
    if np.any(mask_val):
        min_val_fold = min(y_fold_val[mask_val].min(), y_fold_val_pred[mask_val].min())
        max_val_fold = max(y_fold_val[mask_val].max(), y_fold_val_pred[mask_val].max())
        plt.plot([min_val_fold, max_val_fold], [min_val_fold, max_val_fold], '--', c='#595959', lw=2)
    else:
        plt.plot([0, 1], [0, 1], '--', c='#595959', lw=2)  # fallback

    textstr_fold = (f'Fold {fold} (Validation set)\nMSE: {val_mse:.4f}\n'
                    f'RMSE: {val_rmse:.4f}\nMAE: {val_mae:.4f}\nR²: {val_r2:.4f}')
    props_fold = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.05, 0.95, textstr_fold, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=props_fold, fontname='Arial')
    plt.xlabel('Actual power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.ylabel('Predicted power density ×10² (kW/m³)', fontsize=14, fontname='Arial')
    plt.title(f'Ridge – Cross-validation Fold {fold}', fontsize=13, fontweight='bold')
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
    predicted = best_ridge.predict(Res)   # pipeline scales automatically
    predicted_t1.append(predicted[0])

for t2 in t_values:
    Res = pd.DataFrame([[t2, 248, 0.28, 816, 2.16, 1.484966, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t2.append(predicted[0])

for t3 in t_values:
    Res = pd.DataFrame([[t3, 187.21, 0.15, 912, 2.29, 1.659668, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t3.append(predicted[0])

for t4 in t_values:
    Res = pd.DataFrame([[t4, 155, 2.35, 300, 1.7, 0.545943, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t4.append(predicted[0])

for t5 in t_values:
    Res = pd.DataFrame([[t5, 169, 0.29, 906, 2.21, 1.648749, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t5.append(predicted[0])

for t6 in t_values:
    Res = pd.DataFrame([[t6, 227.3, 7.2, 1000, 3.22, 1.819811, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t6.append(predicted[0])

for t7 in t_values:
    Res = pd.DataFrame([[t7, 186.7, 0.2, 897, 2.17, 1.632371, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t7.append(predicted[0])

for t8 in t_values:
    Res = pd.DataFrame([[t8, 174, 5.112, 1348, 2.06, 2.453106, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t8.append(predicted[0])

for t9 in t_values:
    Res = pd.DataFrame([[t9, 244.4, 0.6, 1937, 2.9, 3.524975, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t9.append(predicted[0])

for t10 in t_values:
    Res = pd.DataFrame([[t10, 237.6, 0.52, 1345, 1.27, 2.447646, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t10.append(predicted[0])

for t11 in t_values:
    Res = pd.DataFrame([[t11, 169, 0.7, 1450, 1.83, 2.638727, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
    predicted_t11.append(predicted[0])

for t12 in t_values:
    Res = pd.DataFrame([[t12, 333.7, 0.8, 1346, 1.98, 2.449466, 0.046636,
                         0.361195, 0.4, 0.01885, 40, 10, 500]],
                       columns=X_train.columns)
    predicted = best_ridge.predict(Res)
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
