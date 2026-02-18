import pandas as pd
import numpy as np

# # Loading data
df = pd.read_excel(r'F:\PCM.xlsx')
df;
y = df['PD(W/cm3)'] #Use the PD (W/kg) column as the objective function
y;

# Check and locate NaN values
nan_count = y.isna().sum()
print(f"Number of NaN values: {nan_count}")
nan_locations = y[y.isna()]
print("Locations of NaN values:\n", nan_locations)

# Calculate the correlation matrix of descriptors
import seaborn as sns
import matplotlib.pyplot as plt

# Remove unnecessary columns
columns_to_drop = ['Materials', 'ST', 'SC', 'FP', 'Reference']
df_reduced = df.drop(columns=columns_to_drop)

# Create new columns and put the objective function column as the first column.
columns = ['PD(W/cm3)'] + [col for col in df_reduced.columns if col != 'PD(W/cm3)']
df_reduced = df_reduced[columns]

# Create the correlation matrix
correlation_matrix = df_reduced.corr()

# Setting the figure size
plt.figure(figsize=(8, 7.2))

# Plot a heatmap
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1})

# Customize font size and font name for x and y axis labels
ax.set_xlabel('Features', fontsize=14, fontname='Arial')
ax.set_ylabel('Featuress', fontsize=14, fontname='Arial')

# Customize font size and font name for x and y axis tick labels
ax.tick_params(axis='x', labelsize=9, rotation=90)
ax.tick_params(axis='y', labelsize=9, rotation=0)

# Set font for x and y axis tick labels
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

# Customize color bar font size and font name
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')

# Show the plot
plt.tight_layout()
plt.show()

columns_to_drop = ['Materials', 'Density(kg/m3)', 'ST', 'SC', 'FP', 'Reference']
# Drop the specified columns
df_reduced = df.drop(columns=columns_to_drop)

# Rearrange the columns to make 'PD(W/kg)' the first column
columns = ['PD(W/cm3)'] + [col for col in df_reduced.columns if col != 'PD(W/cm3)']
df_reduced = df_reduced[columns]

# Compute the correlation matrix
correlation_matrix = df_reduced.corr()

# Plotting the results
plt.figure(figsize=(8, 7.2))

# Create a heatmap
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1})

# Customize font size and font name for x and y axis labels
ax.set_xlabel('Features', fontsize=14, fontname='Arial')
ax.set_ylabel('Features', fontsize=14, fontname='Arial')

# Customize font size and font name for x and y axis tick labels
ax.tick_params(axis='x', labelsize=9, rotation=90)
ax.tick_params(axis='y', labelsize=9, rotation=0)

# Set font for x and y axis tick labels
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

# Customize color bar font size and font name
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')

# Show the plot
plt.tight_layout()
plt.show()

# Descriptors, training and testing data set
X = df.drop(['PD(W/cm3)', 'Materials', 'Density(kg/m3)', 'ST', 'SC', 'FP', 'Reference'], axis=1)
X;

# Check for NaN values
nan_count = X.isna().sum()
print(f"Number of NaN values: {nan_count}")

# Locations of NaN values
nan_locations = X[y.isna()]
print("Locations of NaN values:\n", nan_locations)

# ### Data splitting: Training: 70% , Testing: 30%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# to see the X_train, remove ;
X_train;

X_test;

import pandas as pd

# To display X_test with the original indices
X_test_with_indices = X_test.reset_index()
X_test_with_indices.head()

X_test_with_indices;

# Set display options to show all rows
pd.set_option('display.max_rows', None)
X_test_with_indices;


# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(bootstrap=True,
                           max_depth=20,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           n_estimators=500,
                           random_state=100
                           )
rf.fit(X_train, y_train)
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Calculate metrics
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_rmse = np.sqrt(rf_train_mse)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_train_mae = mean_absolute_error(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_rmse = np.sqrt(rf_test_mse)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
rf_test_mae = mean_absolute_error(y_test, y_rf_test_pred)

# Create a DataFrame to store the results with RMSE included
rf_results = pd.DataFrame([['Random Forest', rf_train_mse, rf_train_rmse, rf_train_mae, rf_train_r2,
                            rf_test_mse, rf_test_rmse, rf_test_mae, rf_test_r2]],
                          columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
                                   'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

rf_results


import matplotlib.pyplot as plt
import numpy as np

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 30  # Set the dot size (adjust as needed)
plt.scatter(x=y_train, y=y_rf_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_rf_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Title with custom font properties
# plt.title('Random Forest: Training', fontsize=12, fontweight='bold', fontname='Times New Roman')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.2)

# Add a box with MSE and R² values
textstr = f'RF: Training\nRMSE: {rf_train_rmse:.4f}\nMAE: {rf_train_mae:.4f}\nR$^2$: {rf_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.tight_layout()
plt.show()


# Create the figure
plt.figure(figsize=(6, 5))
# Scatter plot with controllable dot size
dot_size = 30
plt.scatter(x=y_test, y=y_rf_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_rf_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.2)

# Add a box with MSE, MAE and R² values
textstr = f'RF: Testing\nRMSE: {rf_test_rmse:.4f}\nMAE: {rf_test_mae:.4f}\nR$^2$: {rf_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

# Feature importance
feature_importances = rf.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances with customizations
plt.figure(figsize=(6.4, 4.6))
# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

# Customize the bar colors (you can set different colors for each bar if needed)
for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.2)

# Customize x-tick values
plt.xticks([0, 0.05, 0.1, 0.15, 0.2], fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()
# Show the plot
plt.tight_layout()
plt.show()


train_r2 = r2_score(y_train, y_rf_train_pred)
test_r2 = r2_score(y_test, y_rf_test_pred)

plt.figure(figsize=(5.35, 5))

plt.scatter(y_train, y_rf_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_rf_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')

plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.58, 0.96),
           borderpad=0.4,
           labelspacing=0.5,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()



# pip install xgboost
# XGBoost
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
xgb_model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.6, reg_lambda=1.7,
                             gamma=0.2,
                             reg_alpha=0.1, random_state=100)
xgb_model.fit(X_train1, y_train)
# Make predictions
y_xgb_train_pred = xgb_model.predict(X_train1)
y_xgb_test_pred = xgb_model.predict(X_test1)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate the model
xgb_train_mse = mean_squared_error(y_train, y_xgb_train_pred)
xgb_train_rmse = np.sqrt(xgb_train_mse)
xgb_train_r2 = r2_score(y_train, y_xgb_train_pred)
xgb_train_mae = mean_absolute_error(y_train, y_xgb_train_pred)

xgb_test_mse = mean_squared_error(y_test, y_xgb_test_pred)
xgb_test_rmse = np.sqrt(xgb_test_mse)
xgb_test_r2 = r2_score(y_test, y_xgb_test_pred)
xgb_test_mae = mean_absolute_error(y_test, y_xgb_test_pred)

# Store results in a DataFrame with RMSE included
xgb_results = pd.DataFrame([['XGBoost', xgb_train_mse, xgb_train_rmse, xgb_train_mae, xgb_train_r2,
                             xgb_test_mse, xgb_test_rmse, xgb_test_mae, xgb_test_r2]],
                           columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
                                    'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display the results
xgb_results


df_models = pd.concat([rf_results, xgb_results], axis=0)
df_models.reset_index(drop=True)

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 30
plt.scatter(x=y_train, y=y_xgb_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_xgb_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.2)

# Add a box with MSE and R² values
textstr = f'XGBoost: Training\nMSE: {xgb_train_mse:.4f}\nMAE: {xgb_train_mae:.4f}\nR$^2$: {xgb_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')

# Show the plot
plt.tight_layout()
plt.show()

# Create the figure
plt.figure(figsize=(6, 5))
# Scatter plot with controllable dot size
dot_size = 30
plt.scatter(x=y_test, y=y_xgb_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_xgb_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.2)

# Add a box with MSE and R² values
textstr = f'XGBoost: Testing\nRMSE: {xgb_test_rmse:.4f}\nMAE: {xgb_test_mae:.4f}\nR$^2$: {xgb_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Feature importance
feature_importances = xgb_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(6.4, 4.6))

# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

# Customize the bar colors (you can set different colors for each bar if needed)
for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.5)

# Customize x-tick values
plt.xticks([0, 0.1, 0.2, 0.3], fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()
# Show the plot
plt.tight_layout()
plt.show()

train_r2 = r2_score(y_train, y_xgb_train_pred)
test_r2 = r2_score(y_test, y_xgb_test_pred)

plt.figure(figsize=(5.35, 5))

plt.scatter(y_train, y_xgb_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_xgb_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$))', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')

plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.58, 0.96),
           borderpad=0.4,
           labelspacing=0.6,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()


# Extra Trees
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the Extra Trees Regressor model
etr_model = ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=100)

# Train the model
etr_model.fit(X_train_scaled, y_train)

# Make predictions
y_etr_train_pred = etr_model.predict(X_train_scaled)
y_etr_test_pred = etr_model.predict(X_test_scaled)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate the model
etr_train_mse = mean_squared_error(y_train, y_etr_train_pred)
etr_train_rmse = np.sqrt(etr_train_mse)
etr_train_r2 = r2_score(y_train, y_etr_train_pred)
etr_train_mae = mean_absolute_error(y_train, y_etr_train_pred)

etr_test_mse = mean_squared_error(y_test, y_etr_test_pred)
etr_test_rmse = np.sqrt(etr_test_mse)
etr_test_r2 = r2_score(y_test, y_etr_test_pred)
etr_test_mae = mean_absolute_error(y_test, y_etr_test_pred)

# Store results in a DataFrame with RMSE included
etr_results = pd.DataFrame([['Extra Trees', etr_train_mse, etr_train_rmse, etr_train_mae, etr_train_r2,
                             etr_test_mse, etr_test_rmse, etr_test_mae, etr_test_r2]],
                           columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
                                    'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display the results
etr_results


# Concatenate results with other models' results
df_models = pd.concat([rf_results, xgb_results, etr_results], axis=0)
df_models.reset_index(drop=True, inplace=True)
# print(df_models)
df_models.reset_index(drop=True)

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 30  # Set the dot size (adjust as needed)
plt.scatter(x=y_train, y=y_etr_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_etr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'ETR: Training\nMSE: {etr_train_mse:.4f}\nMAE: {etr_train_mae:.4f}\nR$^2$: {etr_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 30
plt.scatter(x=y_test, y=y_etr_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_etr_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'ETR: Testing\nRMSE: {etr_test_rmse:.4f}\nMAE: {etr_test_mae:.4f}\nR$^2$: {etr_test_r2:.5f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.tight_layout()
plt.show()

# Feature importance
etr_feature_importances = etr_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': etr_feature_importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(6.4, 4.6))

# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

# Customize the bar colors (you can set different colors for each bar if needed)
for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.5)

# Customize x-tick values
plt.xticks([0, 0.05, 0.1, 0.15, 0.2], fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()
# Show the plot
plt.tight_layout()
plt.show()


train_r2 = r2_score(y_train, y_etr_train_pred)
test_r2 = r2_score(y_test, y_etr_test_pred)

plt.figure(figsize=(5.3, 5))

plt.scatter(y_train, y_etr_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_etr_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')


plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.58, 0.96),
           borderpad=0.4,
           labelspacing=0.5,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()



# pip install lightgbm
# LightGBM
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


# Optional: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the LightGBM Regressor model
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=10, subsample=1,
                               colsample_bytree=1, reg_alpha=0, random_state=100)

# Train the model
lgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_lgb_train_pred = lgb_model.predict(X_train_scaled)
y_lgb_test_pred = lgb_model.predict(X_test_scaled)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate the model
lgb_train_mse = mean_squared_error(y_train, y_lgb_train_pred)
lgb_train_rmse = np.sqrt(lgb_train_mse)
lgb_train_r2 = r2_score(y_train, y_lgb_train_pred)
lgb_train_mae = mean_absolute_error(y_train, y_lgb_train_pred)

lgbm_test_mse = mean_squared_error(y_test, y_lgb_test_pred)
lgbm_test_rmse = np.sqrt(lgbm_test_mse)
lgbm_test_r2 = r2_score(y_test, y_lgb_test_pred)
lgbm_test_mae = mean_absolute_error(y_test, y_lgb_test_pred)

# Store results in a DataFrame with RMSE included
lgbm_results = pd.DataFrame([['LightGBM', lgb_train_mse, lgb_train_rmse, lgb_train_mae, lgb_train_r2,
                              lgbm_test_mse, lgbm_test_rmse, lgbm_test_mae, lgbm_test_r2]],
                            columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
                                     'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display the results
lgbm_results


# Concatenate results with other models' results
df_models = pd.concat([lgbm_results], axis=0)
df_models.reset_index(drop=True, inplace=True)
# print(df_models)
df_models.reset_index(drop=True)

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 50
plt.scatter(x=y_train, y=y_lgb_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_lgb_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'LGBM: Training\nMSE: {lgb_train_mse:.4f}\nMAE: {lgb_train_mae:.4f}\nR$^2$: {lgb_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Create the figure
plt.figure(figsize=(6, 5))
# Scatter plot with controllable dot size
dot_size = 50
plt.scatter(x=y_test, y=y_lgb_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_lgb_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')
# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'LGBM: Testing\nRMSE: {lgbm_test_rmse:.4f}\nMAE: {lgbm_test_mae:.4f}\nR$^2$: {lgbm_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Feature importance
feature_importances = lgb_model.feature_importances_

# Normalize so that the importances sum to 1
feature_importances_normalized = feature_importances / np.sum(feature_importances)

features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_normalized
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(6.4, 4.6))

# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.5)

# Customize x-tick values
plt.xticks([0, 0.1, 0.2, 0.3], fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()


train_r2 = r2_score(y_train, y_lgb_train_pred)
test_r2 = r2_score(y_test, y_lgb_test_pred)

plt.figure(figsize=(5.4, 5))

plt.scatter(y_train, y_lgb_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_lgb_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')


plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.593, 0.96),
           borderpad=0.4,
           labelspacing=0.5,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()



# pip install catboost
# Cat Boost
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the CatBoost Regressor model
catboost_model = CatBoostRegressor(iterations=2000, learning_rate=0.09, depth=8, l2_leaf_reg=3, subsample=0.1,
                                   max_leaves=31, random_seed=100, verbose=0)

# Train the model
catboost_model.fit(X_train_scaled, y_train)

# Make predictions
y_catboost_train_pred = catboost_model.predict(X_train_scaled)
y_catboost_test_pred = catboost_model.predict(X_test_scaled)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate the model
catboost_train_mse = mean_squared_error(y_train, y_catboost_train_pred)
catboost_train_rmse = np.sqrt(catboost_train_mse)
catboost_train_r2 = r2_score(y_train, y_catboost_train_pred)
catboost_train_mae = mean_absolute_error(y_train, y_catboost_train_pred)

catboost_test_mse = mean_squared_error(y_test, y_catboost_test_pred)
catboost_test_rmse = np.sqrt(catboost_test_mse)
catboost_test_r2 = r2_score(y_test, y_catboost_test_pred)
catboost_test_mae = mean_absolute_error(y_test, y_catboost_test_pred)

# Store results in a DataFrame with RMSE included
catboost_results = pd.DataFrame(
    [['CatBoost', catboost_train_mse, catboost_train_rmse, catboost_train_mae, catboost_train_r2,
      catboost_test_mse, catboost_test_rmse, catboost_test_mae, catboost_test_r2]],
    columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
             'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display the results
catboost_results


# Concatenate results with other models' results
df_models = pd.concat([catboost_results], axis=0)
df_models.reset_index(drop=True, inplace=True)

df_models.reset_index(drop=True)

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 50
plt.scatter(x=y_train, y=y_catboost_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_catboost_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')


# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'CatBoost: Training\nMSE: {catboost_train_mse:.4f}\nMAE: {catboost_train_mae:.4f}\nR$^2$: {catboost_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 50
plt.scatter(x=y_test, y=y_catboost_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_catboost_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'CatBoost: Testing\nRMSE: {catboost_test_rmse:.4f}\nMAE: {catboost_test_mae:.4f}\nR$^2$: {catboost_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

from matplotlib import font_manager

# Feature importance
# Get raw feature importances
feature_importances = catboost_model.feature_importances_
features = X_train.columns

# Normalize so that the importances sum to 1
feature_importances_normalized = feature_importances / np.sum(feature_importances)

features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_normalized
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(6.4, 4.6))

# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

# Customize the bar colors
for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.5)

# Customize x-tick values
plt.xticks(fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()



train_r2 = r2_score(y_train, y_catboost_train_pred)
test_r2 = r2_score(y_test, y_catboost_test_pred)

plt.figure(figsize=(5.355, 5))

plt.scatter(y_train, y_catboost_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_catboost_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')


plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.58, 0.96),
           borderpad=0.4,
           labelspacing=0.5,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()



# Decision Tree
from sklearn.tree import DecisionTreeRegressor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the Decision Tree Regressor model with hyperparameter tuning
dt_model = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=100,
    min_weight_fraction_leaf=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0,

    ccp_alpha=0.0,
    random_state=100
)
# Train the model
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_dt_train_pred = dt_model.predict(X_train_scaled)
y_dt_test_pred = dt_model.predict(X_test_scaled)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate the model
dt_train_mse = mean_squared_error(y_train, y_dt_train_pred)
dt_train_rmse = np.sqrt(dt_train_mse)
dt_train_r2 = r2_score(y_train, y_dt_train_pred)
dt_train_mae = mean_absolute_error(y_train, y_dt_train_pred)

dt_test_mse = mean_squared_error(y_test, y_dt_test_pred)
dt_test_rmse = np.sqrt(dt_test_mse)
dt_test_r2 = r2_score(y_test, y_dt_test_pred)
dt_test_mae = mean_absolute_error(y_test, y_dt_test_pred)

# Store results in a DataFrame with RMSE included
dt_results = pd.DataFrame([['Decision Tree', dt_train_mse, dt_train_rmse, dt_train_mae, dt_train_r2,
                            dt_test_mse, dt_test_rmse, dt_test_mae, dt_test_r2]],
                          columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
                                   'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display the results
dt_results

df_models = pd.concat([dt_results], axis=0)
df_models.reset_index(drop=True, inplace=True)

df_models.reset_index(drop=True)

plt.figure(figsize=(6, 5))

dot_size = 50
plt.scatter(x=y_train, y=y_dt_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_train, y_dt_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

plt.grid(True, linestyle='--', alpha=0.5)

textstr = f'DT: Training\nMSE: {dt_train_mse:.4f}\nMAE: {dt_train_mae:.4f}\nR$^2$: {dt_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.show()

# Create the figure
plt.figure(figsize=(6, 5))

# Scatter plot with controllable dot size
dot_size = 50
plt.scatter(x=y_test, y=y_dt_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Line of best fit
z = np.polyfit(y_test, y_dt_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)

# Labels with custom font properties
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Customize ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add a box with MSE and R² values
textstr = f'DT: Testing\nRMSE: {dt_test_rmse:.4f}\nMAE: {dt_test_mae:.4f}\nR$^2$: {dt_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
plt.show()

from matplotlib import font_manager

# Feature importance
feature_importances = dt_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(6.4, 4.6))

# Customize font properties
font_props = font_manager.FontProperties(family='Arial', size=15, weight='bold')

# Plot the bars
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3aaeff', height=0.7)

for bar in bars:
    bar.set_color('#3aaeff')

# Set font properties for labels
plt.xlabel('Importance', fontsize=15, fontname='Arial')
plt.ylabel('Feature', fontsize=15, fontname='Arial')
plt.grid(True, linestyle='--', alpha=0.5)

# Customize x-tick values
plt.xticks([0, 0.1, 0.2, 0.3], fontsize=15, fontname='Arial', color='black')
plt.yticks(fontsize=10, fontname='Arial', color='black')

# Invert y-axis to have the most important feature at the top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()


train_r2 = r2_score(y_train, y_dt_train_pred)
test_r2 = r2_score(y_test, y_dt_test_pred)

plt.figure(figsize=(5.32, 5))

plt.scatter(y_train, y_dt_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

plt.scatter(y_test, y_dt_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='#595959', lw=2)

plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=15, fontname='Arial')

plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

legend_prop = {'family': 'Arial', 'size': 15}
plt.legend(prop=legend_prop,
           frameon=True,
           fancybox=True,
           framealpha=0.6,
           edgecolor='black',
           facecolor='white',
           bbox_to_anchor=(0.58, 0.96),
           borderpad=0.4,
           labelspacing=0.5,
           handletextpad=0.2,
           borderaxespad=0.2)

plt.grid(True, linestyle='--', alpha=0.5)


plt.axis('equal')
plt.tight_layout()
plt.show()