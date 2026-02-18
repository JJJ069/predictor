import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading data
df = pd.read_excel(r'F:\PCM.xlsx')
df;
y = df['PD(W/cm3)'] #Use the PD (W/cm3) column as the objective
y;

# Check and locate NaN values
nan_count = y.isna().sum()
print(f"Number of NaN values: {nan_count}")
nan_locations = y[y.isna()]
print("Locations of NaN values:\n", nan_locations)

# Remove unnecessary columns
columns_to_drop = ['Materials', 'ST', 'SC', 'FP', 'Reference']
df_reduced = df.drop(columns=columns_to_drop)

# Create new columns and put the objective column as the first column
columns = ['PD(W/cm3)'] + [col for col in df_reduced.columns if col != 'PD(W/cm3)']
df_reduced = df_reduced[columns]

# Create the correlation matrix
correlation_matrix = df_reduced.corr()

plt.figure(figsize=(8, 7.2))

# Plot the correlation heatmap
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1})

ax.set_xlabel('Feature variables', fontsize=15, fontname='Arial')
ax.set_ylabel('Feature variables', fontsize=15, fontname='Arial')

ax.tick_params(axis='x', labelsize=10, rotation=90)
ax.tick_params(axis='y', labelsize=10, rotation=0)

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

# Streamlined features
columns_to_drop = ['Materials', 'Density(kg/m3)', 'ST', 'SC', 'FP', 'Reference']
df_reduced = df.drop(columns=columns_to_drop)

# Ensure 'PD(W/cm3)' is the first column
columns = ['PD(W/cm3)'] + [col for col in df_reduced.columns if col != 'PD(W/cm3)']
df_reduced = df_reduced[columns]

# Compute the correlation matrix
correlation_matrix = df_reduced.corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 7.2))

ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                 annot_kws={"size": 8, "weight": "bold", "fontname": "Arial"},
                 fmt='.2f', cbar_kws={'shrink': 1, 'pad': 0.02})

ax.set_xlabel('Feature variables', fontsize=20, fontname='Arial')
ax.set_ylabel('Feature variables', fontsize=20, fontname='Arial')

ax.tick_params(axis='x', labelsize=12, rotation=90)
ax.tick_params(axis='y', labelsize=12, rotation=0)

# Set font for x and y axis tick labels
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

# Customize color bar font size and font name
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')

# Show the plot
plt.tight_layout()
plt.show()

# Dataset
X = df.drop(['PD(W/cm3)', 'Materials', 'Density(kg/m3)', 'ST', 'SC', 'FP', 'Reference'], axis=1)
X;

# Check for NaN values
nan_count = X.isna().sum()
print(f"Number of NaN values: {nan_count}")

# Locations of NaN values
nan_locations = X[y.isna()]
print("Locations of NaN values:\n", nan_locations)

#Data splitting: 70% for training, 30% for testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


X_train;

X_test;

import pandas as pd

# To display X_test
X_test_with_indices = X_test.reset_index()
# Display the first few rows
X_test_with_indices.head()

X_test_with_indices;

# Set display options to show all rows
pd.set_option('display.max_rows', None)
X_test_with_indices;



# pip install catboost

from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CatBoost Regressor model
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
catboost_train_rmse = np.sqrt(catboost_train_mse)  # Calculate RMSE for training data
catboost_train_r2 = r2_score(y_train, y_catboost_train_pred)
catboost_train_mae = mean_absolute_error(y_train, y_catboost_train_pred)

catboost_test_mse = mean_squared_error(y_test, y_catboost_test_pred)
catboost_test_rmse = np.sqrt(catboost_test_mse)  # Calculate RMSE for testing data
catboost_test_r2 = r2_score(y_test, y_catboost_test_pred)
catboost_test_mae = mean_absolute_error(y_test, y_catboost_test_pred)

# Store results
catboost_results = pd.DataFrame(
    [['CatBoost', catboost_train_mse, catboost_train_rmse, catboost_train_mae, catboost_train_r2,
      catboost_test_mse, catboost_test_rmse, catboost_test_mae, catboost_test_r2]],
    columns=['Method', 'Training MSE', 'Training RMSE', 'Training MAE', 'Training R2',
             'Test MSE', 'Test RMSE', 'Test MAE', 'Test R2'])

# Display results
catboost_results


df_models = pd.concat([catboost_results], axis=0)
df_models.reset_index(drop=True, inplace=True)

df_models.reset_index(drop=True)

# Plot
plt.figure(figsize=(6, 5))

# Data scatter plot
dot_size = 50
plt.scatter(x=y_train, y=y_catboost_train_pred, c='#EC7166', alpha=1, s=dot_size)

# Best-fit line
z = np.polyfit(y_train, y_catboost_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), c='#595959', linestyle=':', linewidth=2)

# Figure labels
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Self-defined ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')


# Add a description box
textstr = f'CatBoost: Training\nRMSE: {catboost_train_mse:.4f}\nMAE: {catboost_train_mae:.4f}\nR$^2$: {catboost_train_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()

# Plot
plt.figure(figsize=(6, 5))

# Data scatter plot
dot_size = 50
plt.scatter(x=y_test, y=y_catboost_test_pred, c='#75C5EB', alpha=1, s=dot_size)

# Best-fit line
z = np.polyfit(y_test, y_catboost_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), c='#595959', linestyle=':', linewidth=2)  # Custom line style and width

# Figure labels
plt.ylabel('Predicted power density (W/cm3)', fontsize=15, fontname='Arial')
plt.xlabel('Actual power density (W/cm3)', fontsize=15, fontname='Arial')

# Self-defined ticks
plt.xticks(fontsize=15, fontname='Arial', rotation=0)
plt.yticks(fontsize=15, fontname='Arial')


# Add a description box
textstr = f'CatBoost: Testing\nRMSE: {catboost_test_rmse:.4f}\nMAE: {catboost_test_mae:.4f}\nR$^2$: {catboost_test_r2:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=15,
         verticalalignment='top', bbox=props, fontname='Arial')
# Show the plot
plt.show()



# Feature importance
feature_importances = catboost_model.feature_importances_
features = X_train.columns

# Normalize to make the importance sum to 1
feature_importances_normalized = feature_importances / np.sum(feature_importances)

features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances_normalized
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7.2, 6.4))

font_props = font_manager.FontProperties(family='Arial', size=17, weight='bold')

# Adjust the color and transparency of 15 columns
colors = [
    (125/255, 198/255, 155/255, 0.7),  # First column: green, translucency 0.5
    (173/255, 97/255, 163/255, 0.6),   # Second column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Third column：grey, translucency 0.8
    (173/255, 97/255, 163/255, 0.6),   # Fourth column：purple, translucency 0.5
    (125/255, 198/255, 155/255, 0.7),  # Fifth column：green, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Sixth column：grey, translucency 0.8
    (173/255, 97/255, 163/255, 0.6),   # Seventh column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Eighth column：grey, translucency 0.8
    (173/255, 97/255, 163/255, 0.6),   # Ninth column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8),  # Tenth column：grey, translucency 0.8
    (125/255, 198/255, 155/255, 0.7),  # Eleventh column：green, translucency 0.5
    (125/255, 198/255, 155/255, 0.7),  # Twelfth column：green, translucency 0.5
    (125/255, 198/255, 155/255, 0.7),  # Thirteenth column：green, translucency 0.5
    (173/255, 97/255, 163/255, 0.6),   # Fourteenth column：purple, translucency 0.5
    (158/255, 170/255, 209/255, 0.8)   # Fifteenth column：grey, translucency 0.8
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




# Plot the testing dataset with the training dataset
train_r2 = r2_score(y_train, y_catboost_train_pred)
test_r2 = r2_score(y_test, y_catboost_test_pred)

plt.figure(figsize=(5.34, 5))

# Plotting the training dataset scatter plot
plt.scatter(y_train, y_catboost_train_pred, alpha=1, c='#EC7166', label=f'Train (R² = {train_r2:.2f})', s=60)

# Plotting the testing dataset scatter plot
plt.scatter(y_test, y_catboost_test_pred, alpha=1, c='#75C5EB', label=f'Test (R² = {test_r2:.2f})', s=60)

# Plot a 45° line
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', c='#595959', lw=2)

# Figure labels
plt.xlabel('Actual power density ($\mathrm{W/cm^3}$)', fontsize=13, fontname='Arial')  # X轴标签
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=13, fontname='Arial')  # Y轴标签

# Self-defined ticks
plt.xticks(fontsize=13, fontname='Arial')
plt.yticks(fontsize=13, fontname='Arial')

# Set the legend
legend_prop = {'family': 'Arial', 'size': 13}
plt.legend(prop=legend_prop,  # Use the 'prop' to set the font family and size
           bbox_to_anchor=(0.45, 0.96),  # Legend position (x, y)
           labelspacing=0.5,  # Vertical spacing between legend items
           handletextpad=0.2,  # Spacing between the legend handle and text
           borderaxespad=0.2,  # The spacing between the legend and the axes
           frameon=False)  # Remove the legend border



ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the same axes to ensure the 45° line
plt.axis('equal')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Predicting the effect of material mass
#t1=Ba(OH)2·8H2O, t2=RT82, t3=Xylitol, t4=MgCl2·6H2O, t5=Molten salt, t6=Erythritol

t_values = np.arange(0, 6, 1)

predicted_t1 = []
predicted_t2 = []
predicted_t3 = []
predicted_t4 = []
predicted_t5 = []
predicted_t6 = []

# Loop over each t value
for t1 in t_values:
    Res = pd.DataFrame([[244.2, 80, 0.6, 2.9, t1, 0, 0, 41.2, 60, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t1.append(predicted_Responsivity[0])

for t2 in t_values:
    Res = pd.DataFrame([[176, 82, 0.2, 2, t2, 0, 0, 41.2, 62, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t2.append(predicted_Responsivity[0])

for t3 in t_values:
    Res = pd.DataFrame([[237.6, 90, 0.52, 1.27, t3, 0, 0, 41.2, 70, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t3.append(predicted_Responsivity[0])

for t4 in t_values:
    Res = pd.DataFrame([[169, 116, 0.7, 1.83, t4, 0, 0, 41.2, 96, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t4.append(predicted_Responsivity[0])

for t5 in t_values:
    Res = pd.DataFrame([[121, 117, 0.734, 2.32, t5, 0, 0, 41.2, 97, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t5.append(predicted_Responsivity[0])

for t6 in t_values:
    Res = pd.DataFrame([[333.7, 118.8, 0.8, 1.98, t6, 0, 0, 41.2, 98.8, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_t6.append(predicted_Responsivity[0])


# Set legend font
legend_font_name = 'Arial'
legend_font_size = 16
legend_title = None

# Set legend location
legend_location = 'upper left'
legend_x_pos = 0.5
legend_y_pos = 0.988

# Set legend style
legend_frame_on = True
legend_fancybox = True
legend_borderpad = 0.4
legend_frame_alpha = 1.0
legend_edgecolor = 'black'

# Plot
plt.figure(figsize=(7, 6))

# Plot data
plt.plot(t_values, predicted_t1, c='#71b7ed', label=r'$\mathrm{Ba(OH)_2\cdot8H_2O}$', marker='o', linewidth=2, markersize=8)
plt.plot(t_values, predicted_t2, c='#b8aeeb', label='RT82', marker='p', linewidth=2, markersize=9)
plt.plot(t_values, predicted_t3, c='#84c3b7', label='Xylitol', marker='D', linewidth=2, markersize=8)
plt.plot(t_values, predicted_t4, c='#88d8db', label=r'$\mathrm{MgCl_2\cdot6H_2O}$', marker='^', linewidth=2, markersize=8)
plt.plot(t_values, predicted_t5, c='#f2b56f', label='Molten salt', marker='v', linewidth=2, markersize=8)
plt.plot(t_values, predicted_t6, c='#f57c6e', label='Erythritol', marker='s', linewidth=2, markersize=8)


# Set axis labels
plt.xlabel('Mass (kg)', fontsize=17, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=17, fontname='Arial')
axis_linewidth = 1.5
tick_linewidth = 1.5
plt.xlim(0, max(t_values))
plt.xticks(np.arange(0, 6, 1), fontsize=20)
plt.yticks(fontsize=20, fontname='Arial')

plt.xlabel('PCM mass (kg)', fontsize=18, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=18, fontname='Arial')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(axis_linewidth)
ax.tick_params(axis='both', which='major', width=tick_linewidth, length=6)


# Create legend
legend = plt.legend(
    loc=legend_location,                    # Legend location
    bbox_to_anchor=(legend_x_pos, legend_y_pos),  # Coordinate of the legend in the plot
    fontsize=legend_font_size,              # Legend font size
    title=legend_title,                     # Figure caption
    frameon=legend_frame_on,                # Show border
    fancybox=legend_fancybox,               # Rounded corner
    borderpad=legend_borderpad,             # Inner margin of the border
    framealpha=legend_frame_alpha,          # Legend background transparency
    edgecolor=legend_edgecolor              # Figure border color
)

plt.tight_layout()
plt.show()


# Predicting the effect of temperature difference
#t7=Ba(OH)2·8H2O, t8=RT82, t9=Xylitol, t10=MgCl2·6H2O, t11=Molten salt, t12=Erythritol

f_values = np.arange(15, 75, 10)

predicted_f7 = []
predicted_f8 = []
predicted_f9 = []
predicted_f10 = []
predicted_f11 = []
predicted_f12 = []

# Loop over each t value
for f7 in f_values:
    Res = pd.DataFrame([[244.2, 80, 0.6, 2.9, 60, 0, 0, f7, 60, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f7.append(predicted_Responsivity[0])

for f8 in f_values:
    Res = pd.DataFrame([[176, 82, 0.2, 2, 60, 0, 0, f8, 62, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f8.append(predicted_Responsivity[0])

for f9 in f_values:
    Res = pd.DataFrame([[237.6, 90, 0.52, 1.27, 60, 0, 0, f9, 70, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f9.append(predicted_Responsivity[0])

for f10 in f_values:
    Res = pd.DataFrame([[169, 116, 0.7, 1.83, 60, 0, 0, f10, 96, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f10.append(predicted_Responsivity[0])

for f11 in f_values:
    Res = pd.DataFrame([[121, 117, 0.734, 2.32, 60, 0, 0, f11, 97, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f11.append(predicted_Responsivity[0])

for f12 in f_values:
    Res = pd.DataFrame([[333.7, 118.8, 0.8, 1.98, 60, 0, 0, f12, 98.8, 0.5655, 400, 0.125, 300, 0.0815, 0]],
                       columns=X_train.columns)
    New_Res = scaler.transform(Res)
    predicted_Responsivity = catboost_model.predict(New_Res)
    predicted_f12.append(predicted_Responsivity[0])


# Set legend font
legend_font_name = 'Arial'
legend_font_size = 16
legend_title = None

# Set legend location
legend_location = 'upper left'
legend_x_pos = 0.5
legend_y_pos = 0.445

# Set legend style
legend_frame_on = True
legend_fancybox = True
legend_borderpad = 0.4
legend_frame_alpha = 1.0
legend_edgecolor = 'black'

# Plot
plt.figure(figsize=(7, 6))

# Plot data
plt.plot(f_values, predicted_f7, c='#71b7ed', label=r'$\mathrm{Ba(OH)_2\cdot8H_2O}$', marker='o', linewidth=2, markersize=8)
plt.plot(f_values, predicted_f8, c='#b8aeeb', label='RT82', marker='p', linewidth=2, markersize=9)
plt.plot(f_values, predicted_f9, c='#84c3b7', label='Xylitol', marker='D', linewidth=2, markersize=8)
plt.plot(f_values, predicted_f10, c='#88d8db', label=r'$\mathrm{MgCl_2\cdot6H_2O}$', marker='^', linewidth=2, markersize=8)
plt.plot(f_values, predicted_f11, c='#f2b56f', label='Molten salt', marker='v', linewidth=2, markersize=8)
plt.plot(f_values, predicted_f12, c='#f57c6e', label='Erythritol', marker='s', linewidth=2, markersize=8)


# Self-defined ticks
plt.xlabel('Mass(kg)', fontsize=17, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=17, fontname='Arial')
axis_linewidth = 1.5
tick_linewidth = 1.5
plt.xlim(20, max(f_values))
plt.xticks(np.arange(15, 75, 10), fontsize=18)
plt.yticks(fontsize=20, fontname='Arial')

plt.xlabel('Temperature difference (°C)', fontsize=18, fontname='Arial')
plt.ylabel('Predicted power density ($\mathrm{W/cm^3}$)', fontsize=18, fontname='Arial')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(axis_linewidth)
ax.tick_params(axis='both', which='major', width=tick_linewidth, length=6)


# Create legend
legend = plt.legend(
    loc=legend_location,
    bbox_to_anchor=(legend_x_pos, legend_y_pos),
    fontsize=legend_font_size,
    title=legend_title,
    frameon=legend_frame_on,
    fancybox=legend_fancybox,
    borderpad=legend_borderpad,
    framealpha=legend_frame_alpha,
    edgecolor=legend_edgecolor
)

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


# Setup for dense grid
x_values_dense = np.linspace(0, 4, 500)  # Specific heat capacity
y_values_dense = np.linspace(0, 400, 500)  # Latent heat
X_dense, Y_dense = np.meshgrid(x_values_dense, y_values_dense)

# Batch prediction
x_flat = X_dense.ravel()
y_flat = Y_dense.ravel()

# Other values remain fixed
fixed_data = [244.2, 80, 0.6, 2.9, 60, 0, 0, 41.2, 60, 0.5655, 400, 0.125, 300, 0.0815, 0]

# Density-intensive data
all_data = []
for x, y in zip(x_flat, y_flat):
    d = fixed_data.copy()
    d[3] = x   # Specific heat capacity
    d[0] = y  # Latent heat
    all_data.append(d)

# Create DataFrame
all_data_df = pd.DataFrame(all_data, columns=X_train.columns)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
all_data_scaled = scaler.transform(all_data_df)

# Predicting and reshaping
Z_dense_flat = catboost_model.predict(all_data_scaled)
Z_dense = Z_dense_flat.reshape(Y_dense.shape)

# Application of Gauss smoothing
Z_dense = gaussian_filter(Z_dense, sigma=sigma_smooth)

plt.figure(figsize=(6, 5))
plt.pcolormesh(X_dense, Y_dense, Z_dense, shading='nearest', cmap='coolwarm')

# Colorbar
cbar = plt.colorbar(pad=0.02)
cbar.set_label('Predicted power density ($\mathrm{W/cm^3}$)', fontname='Arial', fontsize=13)
cbar.ax.tick_params(labelsize=13)
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Self-defined ticks
plt.xticks([0, 1, 2, 3, 4], fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.xlabel('Specific heat capacity (kJ/kgK)', fontname='Arial', fontsize=15)
plt.ylabel(r'Latent heat (kJ/kg)', fontname='Arial', fontsize=15)

# Add Ba(OH)2·8H2O point
# Create a scatter point
plt.scatter(2.9, 244.2, color='#1C3885', marker='x', s=50, alpha=1, label='Point 1')

plt.text(
    0.95,
    0.58,
    r'$\mathrm{Ba(OH)_2\cdot8H_2O}$',
    transform=plt.gca().transAxes,
    color='#1C3885',
    fontsize=12,
    fontfamily='Arial',
    ha='right',
    va='top',
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor='white',
        edgecolor='white',
        alpha=0.1
    )
)
# Plot
plt.tight_layout()
plt.show()



# Heatmap of Temperature difference and Litres per hour
# Setup for dense grid
x_values_dense = np.linspace(10, 50, 500)  # Temperature difference
y_values_dense = np.linspace(100, 350, 500)  # Litres per hour
X_dense, Y_dense = np.meshgrid(x_values_dense, y_values_dense)

# Batch prediction
x_flat = X_dense.ravel()
y_flat = Y_dense.ravel()

# Other values remain fixed
fixed_data = [333.7, 118.8, 0.8, 1.98, 60, 0, 0, 41.2, 98.8, 0.5655, 400, 0.125, 300, 0.0815, 0]

# Density-intensive data
all_data = []
for x, y in zip(x_flat, y_flat):
    d = fixed_data.copy()
    d[7] = x   # Temperature potential
    d[12] = y  # Litres per hour
    all_data.append(d)

# Create DataFrame
all_data_df = pd.DataFrame(all_data, columns=X_train.columns)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
all_data_scaled = scaler.transform(all_data_df)

# Predicting and reshaping
Z_dense_flat = catboost_model.predict(all_data_scaled)
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
plt.xticks([10, 20, 30, 40, 50], fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.xlabel('Temperature difference (°C)', fontname='Arial', fontsize=15)
plt.ylabel(r'Fluid flow rate (L/h)', fontname='Arial', fontsize=15)

# Add Erythritol point
# Create a scatter point
plt.scatter(41.2, 300, color='#F35556', marker='o', s=50, alpha=1, label='Point 1')

plt.text(
    0.89,
    0.875,
    r'Erythritol',
    transform=plt.gca().transAxes,
    color='#F35556',
    fontsize=13,
    fontfamily='Arial',
    ha='right',
    va='top',
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor='white',
        edgecolor='white',
        alpha=0.1
    )
)

# Plot
plt.tight_layout()
plt.show()


# SHAP interpretability analysis
import shap
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Arial'

BAR_COLOR = '#3aaeff'

# SHAP summary plot
SUMMARY_FONTS = {
    'xlabel': 14,
    'ylabel': 13,
    'xtick': 12,
    'ytick': 12,
}

# SHAP bar plot (importance)
BAR_FONTS = {
    'xlabel': 14,
    'ylabel': 12,
    'xtick': 12,
    'ytick': 12,
}

# Create a SHAP interpreter
explainer = shap.TreeExplainer(catboost_model)

# Calculate the SHAP values for the training set
shap_values = explainer.shap_values(X_train_scaled)


plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_scaled,
                  feature_names=X_train.columns.tolist(),
                  show=False,
                  max_display=15)

ax1 = plt.gca()

plt.xlabel('SHAP value (impact on model output)',
          fontsize=SUMMARY_FONTS['xlabel'])

for label in ax1.get_xticklabels():
    label.set_fontsize(SUMMARY_FONTS['xtick'])

for label in ax1.get_yticklabels():
    label.set_fontsize(SUMMARY_FONTS['ytick'])

if ax1.get_ylabel():
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=SUMMARY_FONTS['ylabel'])

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

shap.summary_plot(shap_values, X_train_scaled,
                  feature_names=X_train.columns.tolist(),
                  plot_type="bar",
                  show=False,
                  max_display=15)
ax2 = plt.gca()
plt.xlabel('Average impact on model output magnitude)',
          fontsize=BAR_FONTS['xlabel'])
for patch in ax2.patches:
    patch.set_facecolor(BAR_COLOR)
    patch.set_edgecolor(BAR_COLOR)
for label in ax2.get_xticklabels():
    label.set_fontsize(BAR_FONTS['xtick'])
for label in ax2.get_yticklabels():
    label.set_fontsize(BAR_FONTS['ytick'])
if ax2.get_ylabel():
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=BAR_FONTS['ylabel'])

plt.tight_layout()
plt.show()




