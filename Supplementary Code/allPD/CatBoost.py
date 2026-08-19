import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor
import matplotlib.font_manager as font_manager


# Load data
df = pd.read_excel(r'F:\PCM.xlsx', sheet_name='3600')
y = df['PD×102(kW/m3)']   # Target variable

# First correlation heatmap (all features)
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


# Second correlation heatmap (after dropping more columns)
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

# Prepare feature matrix X (descriptors)
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

# Standardization (new step)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep as DataFrames to preserve column names and indices
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns,
                              index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns,
                             index=X_test.index)

# Define monotonic constraints based on physical rules
# Feature names (must match X_train.columns order)
feature_names = X_train.columns.tolist()

# Map feature names to direction: -1 (decreasing), 1 (increasing), 0 (no constraint)
monotonic_dict = {
    'HCT': -1,    # The predicted value decreases as HCT increases
    'TC': 1,      # The predicted value increases as TC increases
    'Mass': -1,   # The predicted value decreases as Mass increases
    'FVR': 1,     # The predicted value increases as FVR increases
    'ES': 1,      # The predicted value increases as ES increases
    'HTA': 1,     # The predicted value increases as HTA increases
    'TH': 1,      # The predicted value increases as TH increases
    'LPH': 1      # The predicted value increases as LPH increases
}

# Build constraint list in the same order as feature_names
monotone_constraints = []
for feat in feature_names:
    if feat in monotonic_dict:
        monotone_constraints.append(monotonic_dict[feat])
    else:
        monotone_constraints.append(0)   # no constraint for other features

print("Monotone constraints (order matches features):", monotone_constraints)

# CatBoost hyperparameter optimization (using Optuna with TPE)
# Note: monotone constraints are NOT applied during hyperparameter search,
# they will be applied when training the final model and in CV.

def objective(trial):
    iterations = trial.suggest_int('iterations', 100, 1000, step=50)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    depth = trial.suggest_int('depth', 3, 10)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.6, 1.0)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        colsample_bylevel=colsample_bylevel,
        random_seed=100,
        verbose=0,
        thread_count=-1
        # No monotone_constraints here (they are not needed for hyperparameter search)
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=100)
    neg_rmse = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -neg_rmse.mean()   # minimize RMSE

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=100))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)

# Train final model on scaled training data WITH monotonic constraints
catboost = CatBoostRegressor(
    **study.best_params,
    random_seed=100,
    verbose=0,
    thread_count=-1,
    monotone_constraints=monotone_constraints   # Add physical constraints
)
catboost.fit(X_train_scaled, y_train)


# Predictions
y_catboost_train_pred = catboost.predict(X_train_scaled)
y_catboost_test_pred = catboost.predict(X_test_scaled)

# Evaluation metrics
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

# Scatter plots
dot_size = 30

# Training dataset
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

# Testing dataset
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



# Violin plot of residuals for train and test sets
# Calculate residuals (actual - predicted)
train_residuals = y_train - y_catboost_train_pred
test_residuals = y_test - y_catboost_test_pred

# Combine into a DataFrame for seaborn
residual_df = pd.DataFrame({
    'Residual': np.concatenate([train_residuals, test_residuals]),
    'Dataset': ['Train'] * len(train_residuals) + ['Test'] * len(test_residuals)
})

# Create violin plot with no inner lines, only horizontal grid, y-axis range fixed
plt.figure(figsize=(6, 5))
# Fixed FutureWarning by using hue and legend=False
sns.violinplot(x='Dataset', y='Residual', data=residual_df,
               hue='Dataset', palette={'Train': '#EC7166', 'Test': '#75C5EB'},
               inner=None, linewidth=1, legend=False)

# Add zero reference line with low zorder (behind violin)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=0)

plt.xlabel('Dataset', fontsize=15, fontname='Arial')
plt.ylabel('Residual (Actual - Predicted)', fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.yticks([-0.1, 0, 0.1])          # Only show these three tick labels
plt.ylim(-0.1, 0.1)                # Fix y-axis range
plt.grid(axis='y', linestyle='--', alpha=0.3)   # Horizontal grid only
plt.tight_layout()
plt.show()




# Feature importance
# Feature importance
# FIXED: changed catboost_model to catboost
feature_importances = catboost.feature_importances_
features = X_train.columns

# Normalize to make the importance sum to 1
feature_importances_normalized = feature_importances / np.sum(feature_importances)

features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_normalized
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7.6, 6.4))

font_props = font_manager.FontProperties(family='Arial', size=17, weight='bold')

# Adjust the color and transparency of 15 columns
colors = [
    (125/255, 198/255, 155/255, 0.7),  # First column: green, translucency 0.5
    (255/255, 146/255, 172/255, 1),    # Second column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Third column：grey, translucency 0.8
    (158/255, 170/255, 209/255, 0.8),  # Fourth column：grey, translucency 0.8
    (125/255, 198/255, 155/255, 0.7),  # Fifth column：green, translucency 0.5
    (173/255, 97/255, 163/255, 0.6),   # Sixth column：purple, translucency 0.5
    (173/255, 97/255, 163/255, 0.6),   # Seventh column：purple, translucency 0.5
    (173/255, 97/255, 163/255, 0.6),   # Eighth column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Ninth column：grey, translucency 0.8
    (173/255, 97/255, 163/255, 0.6),   # Tenth column：purple, translucency 0.5
    (125/255, 198/255, 155/255, 0.7),  # Eleventh column：grey, translucency 0.8
    (125/255, 198/255, 155/255, 0.7),  # Twelfth column：green, translucency 0.5
    (125/255, 198/255, 155/255, 0.7),  # Thirteenth column：green, translucency 0.5
]


# If the feature number is less than 15, use only the top n colors.
n_features = len(importance_df)
if n_features < 15:
    colors = colors[:n_features]
elif n_features > 15:

# If the feature number exceeds 15, expand the color list.
    import random
    base_colors = colors
    colors = []
    for i in range(n_features):
        if i < 15:
            colors.append(base_colors[i])
        else:
            # Use random color and transparency for additional features
            r = random.randint(0, 255) / 255
            g = random.randint(0, 255) / 255
            b = random.randint(0, 255) / 255
            alpha = random.uniform(0.1, 0.9)
            colors.append((r, g, b, alpha))

# Plot column plot, using different colors and transparency for each column
for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
    plt.barh(feature, importance, height=0.7, color=colors[i])

# Font properties
plt.xlabel('Importance level', fontsize=17, fontname='Arial')
plt.ylabel('Feature variables', fontsize=17, fontname='Arial')


# Self-defined ticks
plt.xticks(fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=12, fontname='Arial', color='black')

# The most important feature is located at the top
plt.gca().invert_yaxis()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Show the plot
plt.tight_layout()
plt.show()

# Feature importance —— add value labels on the right side of bars
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

# Add numeric values at the right edge of each bar
for bar in bars:
    width = bar.get_width()                      # importance value
    x = bar.get_x() + width                      # x-coordinate of bar's right edge
    y = bar.get_y() + bar.get_height() / 2       # y-coordinate at bar center
    # Display with 4 decimal places (adjustable)
    plt.text(x, y, f'{width:.4f}',
             ha='left', va='center', fontsize=8,
             fontname='Arial', color='black')

plt.tight_layout()
plt.show()


# 5‑fold cross‑validation on training set (using scaled data) WITH constraints
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
        **study.best_params,
        random_seed=100,
        verbose=0,
        thread_count=-1,
        monotone_constraints=monotone_constraints   # Add constraints
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

# print results
print(df_results.to_csv(sep='\t', index=False, float_format='%.6f'))






# SHAP interpretability analysis
import shap
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

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
plt.xlabel('SHAP value (impact on model output)', fontsize=14)
for label in ax1.get_xticklabels():
    label.set_fontsize(12)
for label in ax1.get_yticklabels():
    label.set_fontsize(12)
if ax1.get_ylabel():
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=13)
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