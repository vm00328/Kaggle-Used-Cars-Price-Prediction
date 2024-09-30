# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="hlTVezAPUDJN"
# # WORKFLOW
#
# ## Step 1: Importing Libraries & The Original Data
#
# ---
#
# ## Step 2: Exploratory Data Analysis: Understanding the Original Data
#
# **We will analyse and plot:**
#
# 2.1.   the number of cars per brand to get a feeling of the distirbution of the dataset under the *'brand'* column
#
# 2.2.   the distribution of the model year
#
# 2.3.   the distribution of the milage
#
# 2.4.   the distribution of prices
#
# 2.5.   the various fuel types, engines, and transmissions
#
# 2.6.   the vehicles with a reported accident vs those that have not reported one
#
# ---
#
# ## Step 3: Data Cleaning, Pre-Processing & Feature Engineering
#
# 3.1   removal of redundant features following the EDA
#
# 3.2   dropping records with limited statistical importance while ensuring this does not result in a certain class being underrepresented, subsequently leading to an imbalanced dataset
#
# 3.3. Missing data under the Engine Volume, HP, and/or Cylinders columns and the car is electric, the former and the latter will be set to 0.
#
# 3.4. We drop the 'clean_title' column since it contains a single unique value across all records and hence adds no value
#
# 3.5. We drop the interior and exterior colour columns as they are believed to not add any statistical significance and are irrelevant given the project scope
#
# 3.6. categorical brackets for some numeric data to decrease dimensionality
#
# 3.7. We create new columns "Engine Volume", "HP", "Cylinders" by using the available data under the 'engine' column where that is applicable
#
# ---
#
# ## Step 4: Exploring the new features and cleaning if necessary
# -

# !pip install jupytext


jupytext --set-formats ipynb,py your_notebook.ipynb

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="0uoz840Gr84M" outputId="da525f01-6eb1-452e-cb00-4e29ccd65c8b"
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Data Pre-Processing
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Model Evaluation
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)

sns.set(style = 'whitegrid')
sns.set_context('notebook')

data = pd.read_csv('/content/train.csv')

data.head()

# + [markdown] id="uZq_XVFeQ87Y"
# # Step 2: Exploratory Data Analysis
#
# ## 2.1. Brand

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="XBUaGCpF0z8Z" jupyter={"outputs_hidden": true} outputId="784f7a4a-5e70-4d2a-e54f-b6e8d848c922"
# the 20 brands with the most presence in the dataset
print("Number of cars: ", data.shape[0])
print("Number of brands: ", data['brand'].nunique())

sns.countplot(data = data, y = 'brand', order = data['brand'].value_counts().iloc[:20].index)
plt.title('Top 20: Most Cars per Brand')
plt.xlabel('Number of Cars')
plt.ylabel('Brand')
plt.show()

print('\n')

# the 20 brands with the least presence in the dataset
sns.countplot(data = data, y = 'brand', order = data['brand'].value_counts().iloc[-20:].index)
plt.title('Top 20: Fewest Cars per Brand')
plt.xlabel('Number of Cars')
plt.ylabel('Brand')
plt.show()

# + [markdown] id="O8-M5YrBgGO9"
# ## 2.2. Model Year

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="E1wN7AlwgFta" outputId="07f8ced6-bf2e-44e7-fe84-58cb6a50d2ba"
# a histogram of the distirbution of the model year
sns.histplot(data['model_year'], kde = True, bins = 50)
plt.axvline(data['model_year'].mean(), color='red', linestyle='--', label = 'Average Year')
plt.title('Distribution Plot for Model Year')
plt.xlabel('Model Year')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# + [markdown] id="wu598REogJy9"
# ## 2.3. Milage

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="l6jJ-VCxgHX0" outputId="7916f848-8178-4ea7-9e75-4a349022750c"
# average mileage and price for each brand
brand_avg = data.groupby('brand', as_index = False).agg({'milage': 'mean', 'price': 'mean'})

# interactive plot
fig = px.scatter(brand_avg,
                 text='brand',
                 x='milage',
                 y='price',
                 title='Average Milage vs. Price by Brand',
                 labels={'mileage': 'Average Milage', 'price': 'Average Price'})

# marker size and label position
fig.update_traces(marker = dict(size = 4), textposition = 'top right', textfont_size = 9)
fig.show()

# + [markdown] id="z3he-WfkgIlS"
# ## 2.4. Price

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="76VOhqJqgIQ-" outputId="485fd935-5ea3-4656-eb5a-58ffb3ccdbf3"
# a histogram of the distirbution of the price
sns.histplot(data['price'], kde = True, bins = 50)
plt.axvline(data['price'].mean(), color='red', linestyle='--', label = 'Average Price')
plt.title('Distribution Plot for Model Year')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# + [markdown] id="SOFuTxWVgYBW"
# ## 2.5. Price vs Model Year

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="Rp0PdpO1gaGg" outputId="58cf03cf-b238-4ab3-d5b4-97eebfa0cc29"
sns.lineplot(data = data, x = 'model_year', y = 'price')
plt.title('Price vs. Model Year')
plt.xlabel('Model Year')
plt.ylabel('Price')
plt.show()

# + [markdown] id="jgv-70F-TfJy"
# ## 2.6. Fuel

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="_vcbaITz2Wqa" outputId="04d0b8d9-f62d-4c3e-d5dd-d148c78004f9"
# understanding the number of fuels across all vehicles
data.fuel_type.value_counts()

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="OwnfM6TKex_d" outputId="0323df05-0d71-45cc-fa7a-0cd26f7e004d"
# examining the vehicles that have a value 'not supported' under the 'fuel_type' column
data[data['fuel_type'] ==  'not supported'].head()

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="vm79ywt-fzir" outputId="d438ff27-5a1c-41e3-c47d-022ec2e075b9"
data[data['fuel_type'] ==  '–'].head()

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="Eq-MU2OrkwsK" outputId="c9b62790-7457-4f7e-e1a8-8a42e3c2e629"
data[data['fuel_type'] ==  '–']['engine'].value_counts()

# + [markdown] id="qTI2xK4zk3je"
# **Intermediate conclusion:**
#
# *Missing data under the engine column, not necessarily as a NaN but as a '–' results in such under the 'fuel_type' column as well.*

# + [markdown] id="q6gDzo6Ug2Oz"
# ## 2.7. Engine

# + colab={"base_uri": "https://localhost:8080/"} id="oW6mBb9OiBb7" outputId="c1510643-48c5-45a0-d12f-08b50c28ea2b"
print("NaN values under the Engine column: ", data.engine.isna().sum())

# number of records with less than 10 characters under the engine column
print("Number of records with less than 10 characters: ", data[data['engine'].str.len() < 10].shape[0])

# + [markdown] id="u2GbrId1e6CP"
# ## 2.8. Transmission

# + colab={"base_uri": "https://localhost:8080/"} id="9mgZbmf6e5XS" outputId="2140a518-9ff0-49b8-916d-76c30695e6af"
# understanding the number of transmissions across all vehicles
print(data.transmission.value_counts())

# + [markdown] id="dyvBaGaMRnbI"
# # Step 3: Data Cleaning, Pre-Processing & Feature Engineering
#
# ## 3.1. Addressing *fuel_type* data quality issues

# + colab={"base_uri": "https://localhost:8080/", "height": 367} id="QdwG5LtwQ1t6" outputId="3f6bd905-c3ff-459a-a4e0-41906b90e3a6"
# fill under the 'fuel_type' column with the value 'Gasoline' if the word 'Gasoline' is mentioned under the 'engine' column
data.loc[data['engine'].str.contains('Gasoline', case = False), 'fuel_type'] = 'Gasoline'

# fill under the 'fuel_type' column with the value 'Hydrogen' if the word 'Hydrogen' is mentioned under the 'engine' column
data.loc[data['engine'].str.contains('Hydrogen', case = False), 'fuel_type'] = 'Hydrogen'

# drop records under the engine column where the value is '–'
data = data[data['engine'] != '–']

data.fuel_type.value_counts()

# + [markdown] id="ZaF77DVroO4M"
# Addressing the last 2 remaining records labeled as '–' under the 'fuel_type' column

# + colab={"base_uri": "https://localhost:8080/", "height": 112} id="VhSb6ll4nHtA" outputId="0f1293a2-eac8-4379-8c88-ee791e306efa"
data[data['fuel_type'] == '–']

# + colab={"base_uri": "https://localhost:8080/", "height": 53} id="m-NzkQRpnkba" outputId="b093e7e0-63f4-48a1-f230-dbf9d1b2a483"
# make the two remaining records with '–' udner the 'fuel_type' column be 'Gasoline'
data.loc[data['fuel_type'] == '–', 'fuel_type'] = 'Gasoline'
data[data['fuel_type'] == '–']

# + [markdown] id="41HEnq2HoFSM"
# Addressing the last remaining record labeled as 'not supported' under the 'fuel_type' column

# + colab={"base_uri": "https://localhost:8080/", "height": 81} id="BOshU1IDoEU8" outputId="a833e30f-6bd6-4362-8b70-39ff4332c019"
data[data['fuel_type'] == 'not supported']

# + colab={"base_uri": "https://localhost:8080/", "height": 53} id="0oL8enYtoLZA" outputId="566d6f7b-4231-4518-ef49-3056550b81b5"
data.loc[data['fuel_type'] == 'not supported', 'fuel_type'] = 'Gasoline'
data[data['fuel_type'] == 'not supported']

# + colab={"base_uri": "https://localhost:8080/", "height": 321} id="CnfyRqoNsJHZ" outputId="d44670c5-04eb-437f-8ef6-6d91a40cda50"
print("Remaining NaN columns under the 'fuel_type' column: ", data.fuel_type.isna().sum())
data.fuel_type.value_counts()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="nIJOPvEdt4fs" outputId="3186df4f-dfec-4a6d-95ac-9d4ae10a3883"
data[data['fuel_type'].isna()]['engine']

# + [markdown] id="GJk3P8-2tHjn"
# ## 3.2. Dropping redundant columns

# + id="1iexDDJBtMH1"
# dropping redundant columns
data.drop(columns = ['clean_title', 'int_col', 'ext_col'], inplace = True)


data['fuel_type'].fillna('Electric', inplace = True) # filling the missing values in the 'fuel_type' column with 'Electric'
data.loc[(data['brand'] == 'Tesla') & ( (data['fuel_type'] == 'Gasoline') | (data['fuel_type'] == 'Diesel') ), 'fuel_type'] = 'Electric'

# retrieving the HP, engine volume, and cylinders of each vehicle if such information exists under the 'engine' column
data['HP'] = data['engine'].str.extract('(\d+)\.?\d*HP') #fillna('NA')
data['Engine Volume (Liters)'] = data['engine'].str.extract('(\d+\.?\d*)L')
data['Cylinders'] = data['engine'].str.extract('(\d+)\s*Cylinder|V(\d+)').bfill(axis=1).iloc[:, 0]
data.drop(columns = ['engine'], inplace = True)

num_features = ['model_year', 'milage', 'price']
print(f"Dataframe shape: {data.shape}")

print(data.isna().sum(), '\n')

print(data.dtypes, '\n')

data[num_features].describe().apply(lambda s: s.apply('{0:.0f}'.format))

# + [markdown] id="gXxiW5Y0pA5z"
# ## 3.2. Feature Enginnering
#
# ### 3.3. Milage & Price

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="1L9olIoqQixe" outputId="c5848458-5786-4aad-b159-a1678d87b64d"
####################################################### milage & price categorical labels #######################################################
milage_brackets = [100, 25000, 58000, 95000, float('inf')]
milage_labels = ['Low Milage (100-25k)', 'Low-to-Medium Milage (25k-58k)', 'Medium-to-High Milage (58k-95k)', 'High Milage (95k+)']

price_brackets = [2000, 17000, 31000, 50000, float('inf')]
price_labels = ['Low End (2k-17k)', 'Low-to-Medium End (17k-31k)', 'Medium-to-High End (31k-50k)', 'High End (50k+)']

# Apply pd.cut with meaningful labels
data['milage_bracket'] = pd.cut(data['milage'], bins=milage_brackets, labels=milage_labels, include_lowest=True)
data['price_bracket'] = pd.cut(data['price'], bins=price_brackets, labels=price_labels, include_lowest=True)

data.head()


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="srBZs-CGx2xy" outputId="4f0cb2b2-d98c-4f55-de11-51071b8fabdb"
def plot_numeric_features(df, feature):
  plt.figure(figsize = (12, 6))
  sns.histplot(data[feature], kde = True, bins = 50)
  median_price = data[feature].median().astype(int)
  plt.axvline(median_price, color='red', linestyle='--', label = f'Median: {median_price}')
  plt.title(f'Distribution Plot for {feature}')
  plt.xlabel(f'{feature}')
  plt.ylabel('Frequency')
  plt.legend()
  plt.show()

plot_numeric_features(data, 'model_year')
plot_numeric_features(data, 'milage')
plot_numeric_features(data, 'price')

# + [markdown] id="grknC-WC3an_"
# ## Analysing Accidents

# + colab={"base_uri": "https://localhost:8080/", "height": 581} id="Ak3X2ZiLdYL-" outputId="ef8744eb-e8c0-4c47-ef90-bb8dc92085bf"
plt.figure(figsize=(8, 6))
sns.countplot(x='accident', data=data)
plt.title('Accident Distribution')
plt.ylabel('Count')

# + id="oEemO5OfdyWl"
sns.heatmap(data[num_features].corr(), annot = True)


# + [markdown] id="gfTS8bg-4RH9"
# # Data Pre-Processing & Feature Engineering
#
# 1. Handling Outliers
#
# 2. Feature Scaling of numeric features & one-hot encoding of categorical features
#
# 3. Cross Validation
#
# 4. Hyperaparmeter Tuning through GridSearchCV or RandomSearchCV
#
# 5. Fit-transform on train data and only transform on test data
#

# + [markdown] id="wbizbqpTa6Iq"
# ## Outlier Removal

# + id="81zakwVnZw-7"
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_out = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_out

df_train_no_outliers = remove_outliers_iqr(data, 'milage')
df_train_no_outliers = remove_outliers_iqr(df_train_no_outliers, 'price')
df_train_no_outliers.reset_index(drop=True, inplace=True)

# + [markdown] id="pBqqufmmT-rd"
# # Model Training

# + id="Dkgc8xoV7WRq"
# Features and target
X = data.drop(columns = ['price'])
y = data['price']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers = [
        ('num', Pipeline(steps = [
            ('imputer', SimpleImputer(strategy = 'mean')),
            ('scaler', RobustScaler())
        ]), ['model_year', ]),
        ('cat', OneHotEncoder(), ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident'])
    ])

# Creating the model pipeline
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_jobs = -1))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# + colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["1d6ca877f59a4983ad0acf06b1b5937d", "6727b97742fe4e82aaeae644d57e249a", "01ec33bac1bd40c6a74c901898405487", "cfac990ed4a845e3ae567a77a9e8653c", "086bdcdd83b24e2f8b0639c28b6b0550", "1d70a22547f748878b871343f64aef15", "56f30f42421243249b6391567d7d6501", "b3c7a955a2bc41009f53d19124eaa334", "064e8881065b44d6bbcd15e950486c1c", "28e34d7f2f4747e1a2bf5503de8ea5ff", "d6e91344baf74738ab2cc50d6c32eb12"]} id="v0HcxkaJ_iNn" outputId="ad19acbe-738f-42eb-bfde-64ab79eb864f"
# Fitting the model
with tqdm(total=1, desc="Training Model") as pbar:
  model.fit(X_train, y_train)
  pbar.update(1)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# + id="Y3SRFC4zy6YA"
print("Mean Squared Error:", mse)

importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
print("Feature Importance Ranking:")
for idx in sorted_indices:
    print(f"{X.columns[idx]}: {importances[idx]}")

