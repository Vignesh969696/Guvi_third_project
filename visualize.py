# Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv(r"D:\smartpremiums\train.csv")
test_data = pd.read_csv(r"D:\smartpremiums\test.csv")

# Info
print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

print("\nStarting 5 rows of the training set:")
print(train_data.head())

print("\nStatistical summary of numerical features:")
print(train_data.describe())

print("\nData types and counts of non-null values in training data:")
train_data.info()

print("\nFirst 5 rows of the test set:")
print(test_data.head())

print("\nInfo on test set:")
test_data.info()


# Exploratory Data Analysis (EDA)

print("\nMissing values in training data (descending order):")
print(train_data.isnull().sum().sort_values(ascending=False))

print("\nMissing values in test data (descending order):")
print(test_data.isnull().sum().sort_values(ascending=False))

# Separate numerical and categorical columns
num_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = train_data.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical features:")
for col in num_features:
    print(f"- {col}")

print("\nCategorical features:")
for col in cat_features:
    print(f"- {col}")
!pip install --upgrade numpy
!pip install --upgrade pandas scipy scikit-learn matplotlib seaborn xgboost
import numpy as np
import pandas as pd
import sklearn
import xgboost
num_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Plot all numerical features at once
fig = train_data[num_features].hist(
    bins=25,                # slightly fewer bins
    figsize=(16, 10),       # adjusted figure size
    color='skyblue',        # different bar color
    edgecolor='gray'        # slightly different edge color
)

plt.suptitle('Histograms of Numerical Columns', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.show()
# Correlation Heatmap for Numerical Features

import matplotlib.pyplot as plt
import seaborn as sns

num_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr_matrix = train_data[num_features].corr()

plt.figure(figsize=(14, 10))  # making the figure slightly wider
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="RdBu_r",  # different color map
    center=0,       # centering the colormap at 0
    linewidths=0.5  # adding lines between cells
)

plt.title("Numerical Features Correlation", fontsize=16)
plt.tight_layout()
plt.show()
# Count Plots for Categorical Features

cat_features = train_data.select_dtypes(include='object').columns.tolist()

# Plot the first 5 categorical features
for col in cat_features[:5]:
    plt.figure(figsize=(7, 5))  # slightly bigger figure
    sns.countplot(
        x=col,
        data=train_data,
        order=train_data[col].value_counts().index,
        palette="Set2",       # added color palette
        edgecolor='black'     # black borders for bars
    )
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xticks(rotation=30)    # rotating the labels 
    plt.ylabel("Count", fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  #  for addign horizontal gridlines
    plt.tight_layout()
    plt.show()
  # Target Variable Distribution

plt.figure(figsize=(8, 5))
sns.histplot(
    train_data['Premium Amount'], 
    bins=40,             
    kde=True, 
    color='salmon',     
    edgecolor='black'    # black borders for bars
)
plt.title("Distribution of Target Variable: Premium Amount", fontsize=14)
plt.xlabel("Premium Amount", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Skewness of target variable
premium_skew = train_data['Premium Amount'].skew()
print(f"Skewness of Premium Amount: {premium_skew:.4f}")
# Additional Feature Visualizations

plt.figure(figsize=(7, 5))
sns.boxplot(
    x=train_data['Previous Claims'],
    color='lightgreen'
)
plt.title("Boxplot of Previous Claims", fontsize=14)
plt.xlabel("Number of Previous Claims", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    train_data['Health Score'],
    bins=25,          
    kde=True,
    color='skyblue',
    edgecolor='gray'
)
plt.title("Distribution of Health Score", fontsize=14)
plt.xlabel("Health Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    train_data['Annual Income'],
    bins=25,        
    kde=True,
    color='orange',
    edgecolor='black'
)
plt.title("Distribution of Annual Income", fontsize=14)
plt.xlabel("Annual Income", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()
# Numerical Features vs Target Variable

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style and font scale
sns.set(style='ticks', font_scale=1.2)  # changed style for variety

# List of numerical features
num_features = ['Age', 'Annual Income', 'Health Score', 'Previous Claims']

# Plotting each numerical feature against Premium Amount
for col in num_features:
    plt.figure(figsize=(9, 5))  # slightly wider
    
    sns.regplot(
        data=train_data,
        x=col,
        y='Premium Amount',
        scatter_kws={'alpha': 0.4, 's': 35},   #  dot size and transparency
        line_kws={'color': 'green', 'lw': 2.5}, #  line color and width
        ci=90  # confidence interval
    )
    
    plt.title(f"{col} vs Premium Amount", fontsize=14, weight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Premium Amount", fontsize=12)
    plt.grid(True, linestyle='-.', alpha=0.5)  # grid style
    plt.tight_layout()
    plt.show()
  # Target vs Key Features


import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style='whitegrid', font_scale=1.1)

# Categorical Features vs Target
cat_features = ['Gender', 'Marital Status', 'Policy Type', 'Smoking Status']
for col in cat_features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=col, y='Premium Amount', data=train_data, palette='pastel')
    plt.title(f'Premium Amount by {col}', fontsize=13, weight='bold')
    plt.ylabel("Premium Amount")
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Numerical Features vs Target
num_features = ['Age', 'Annual Income', 'Health Score', 'Previous Claims']
for col in num_features:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=train_data[col], y=train_data['Premium Amount'], alpha=0.5, color='teal')
    plt.title(f'{col} vs Premium Amount', fontsize=13, weight='bold')
    plt.xlabel(col)
    plt.ylabel("Premium Amount")
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Outlier Detection in Numerical Features
for col in ['Previous Claims', 'Annual Income', 'Health Score']:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=train_data[col], color='lightcoral')
    plt.title(f'Outlier Check: {col}', fontsize=13)
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
# Date Conversion and Feature Engineering


# Convert 'Policy Start Date' to datetime format in both datasets
for df in [train_data, test_data]:
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')


# Missing Values


# Separating the numerical and categorical columns
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_data.select_dtypes(include=['object']).columns

# Filling missing values in numerical features with median
for col in num_cols:
    median_val = train_data[col].median()
    train_data[col].fillna(median_val, inplace=True)
    if col in test_data.columns:
        test_data[col].fillna(median_val, inplace=True)

# Filling missing values in categorical features with mode
for col in cat_cols:
    mode_val = train_data[col].mode()[0]
    train_data[col].fillna(mode_val, inplace=True)
    if col in test_data.columns:
        test_data[col].fillna(mode_val, inplace=True)

# checking for nulls
print("Missing values after filling (Train):\n", train_data.isnull().sum())
print("Missing values after filling (Test):\n", test_data.isnull().sum())


# Extract date components and policy age


for df in [train_data, test_data]:
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Day'] = df['Policy Start Date'].dt.day
    df['Policy_Age_Days'] = (pd.Timestamp.today() - df['Policy Start Date']).dt.days

# The new date features
print(train_data[['Policy Start Date', 'Policy_Year', 'Policy_Month', 'Policy_Day', 'Policy_Age_Days']].head())
print(test_data[['Policy Start Date', 'Policy_Year', 'Policy_Month', 'Policy_Day', 'Policy_Age_Days']].head())
# Log Transform + Feature Engineering


import numpy as np
import pandas as pd

# Log Transformations 
for df in [train_data, test_data]:
    # Target (Premium Amount) -- ONLY in train, test may not have target
    if 'Premium Amount' in df.columns:
        df['Premium Amount'] = np.log1p(df['Premium Amount'])
    # Annual Income
    df['Annual Income'] = np.log1p(df['Annual Income'])

#  Age Buckets 
age_cutoffs = [18, 30, 45, 60, np.inf]
age_groups = ['18-30', '31-45', '46-60', '60+']
for df in [train_data, test_data]:
    df['Age_Group'] = pd.cut(df['Age'], bins=age_cutoffs, labels=age_groups, include_lowest=True)

# Customer Feedback Encoding
feedback_dict = {'Poor': 0, 'Average': 1, 'Good': 2}
for df in [train_data, test_data]:
    df['Feedback_Score'] = df['Customer Feedback'].map(feedback_dict)

# Income Brackets (use transformed income) 
income_ranges = [0, np.log1p(30_000), np.log1p(60_000), np.log1p(100_000), np.inf]
income_tags = ['Low', 'Medium', 'High', 'Very High']
for df in [train_data, test_data]:
    df['Income_Class'] = pd.cut(df['Annual Income'], bins=income_ranges, labels=income_tags)

# Credit Score Buckets 
credit_limits = [0, 400, 600, 800, np.inf]
for df in [train_data, test_data]:
    df['Credit_Bucket'] = pd.cut(df['Credit Score'], bins=credit_limits, labels=False)

# Dependents Grouping
def categorize_dependents(x):
    if x == 0:
        return "None"
    elif x <= 2:
        return "Few"
    else:
        return "Many"

for df in [train_data, test_data]:
    df['Dependents_Group'] = df['Number of Dependents'].apply(categorize_dependents)

# Policy Duration 
today = pd.Timestamp.now()
for df in [train_data, test_data]:
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['Policy_Duration_Days'] = (today - df['Policy Start Date']).dt.days

# Interaction Features 
for df in [train_data, test_data]:
    df['Age_x_Health'] = df['Age'] * df['Health Score']
    df['Credit_x_Claims'] = df['Credit Score'].fillna(0) * df['Previous Claims'].fillna(0)
    df['Income_x_Credit'] = df['Annual Income'] * df['Credit Score']

# Risk Flags 
for df in [train_data, test_data]:
    df['Smoker_Flag'] = df['Smoking Status'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df['LowCredit_Flag'] = df['Credit Score'].apply(lambda x: 1 if x < 600 else 0)
    df['MultiClaims_Flag'] = df['Previous Claims'].apply(lambda x: 1 if x > 2 else 0)

# Exercise Frequency Encoding 
exercise_dict = {'Never': 0, 'Rarely': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4}
for df in [train_data, test_data]:
    df['Exercise_Score'] = df['Exercise Frequency'].map(exercise_dict)

# Missing Value Check 
print("Remaining nulls in train:\n", train_data.isnull().sum()[lambda x: x > 0])
print("\nRemaining nulls in test:\n", test_data.isnull().sum()[lambda x: x > 0])
# Drop potential leaky features

leaky_cols = ['Customer Feedback', 'Feedback_Score']

for df_name, df in [('train_data', train_data), ('test_data', test_data)]:
    df.drop(columns=leaky_cols, inplace=True, errors='ignore')
    print(f"Dropped {set(leaky_cols).intersection(df.columns)} from {df_name}")
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Silence Git warning
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
np.random.seed(42)

# Target and features
TARGET = 'Premium Amount'
drop_cols = ['id', 'Policy Start Date', 'Customer Feedback', 'Exercise Frequency', 'Smoking Status']
X = train_data.drop(columns=[TARGET] + drop_cols, errors='ignore')
y = train_data[TARGET]

# Convert object columns to category type
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

# Convert int columns with missing values to float
for col in X.select_dtypes(include=['int64', 'int32']).columns:
    if X[col].isna().any():
        X[col] = X[col].astype('float64')


# Subsample data 

sample_size = 200_000
X = X.sample(n=sample_size, random_state=42)
y = y.loc[X.index]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical features
num_features = X.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
cat_features = X.select_dtypes(include=['category']).columns.tolist()

# Pipelines for preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_features),
    ('cat', categorical_pipeline, cat_features)
])

# Model pipelines
model_pipelines = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor),
                                   ('regressor', LinearRegression())]),
    'Decision Tree': Pipeline([('preprocessor', preprocessor),
                               ('regressor', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline([('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    'XGBoost': Pipeline([('preprocessor', preprocessor),
                         ('regressor', XGBRegressor(random_state=42, verbosity=0, n_jobs=-1))])
}

# Randomized search parameters 
param_distributions = {
    'Linear Regression': {},
    'Decision Tree': {'regressor__max_depth': [5, 10],
                      'regressor__min_samples_split': [2, 5]},
    'Random Forest': {'regressor__n_estimators': [20, 40],
                      'regressor__max_depth': [5, 7]},
    'XGBoost': {'regressor__n_estimators': [10, 20],
                'regressor__max_depth': [3, 6]}
}

# Model training & evaluation
results = {}
for model_name, pipeline in model_pipelines.items():
    print(f"\nTraining {model_name} ...")
    search = RandomizedSearchCV(pipeline, param_distributions[model_name],
                                n_iter=5,  
                                cv=3,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1,
                                random_state=42,
                                verbose=0)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    predictions = np.clip(best_model.predict(X_val), 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    print(f"----- {model_name} -----")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    results[model_name] = (best_model, rmse, mae, r2)

# Select best model
best_model_name = min(results, key=lambda x: results[x][1])
best_model = results[best_model_name][0]
print(f"\nBest model: {best_model_name}")
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ignore all warnings
warnings.filterwarnings("ignore")

# Convert categorical columns to string for MLflow
X_log = X_val.copy()  # Use X_val from your fast sampled split
for col in X_log.select_dtypes(include='category').columns:
    X_log[col] = X_log[col].astype(str)

# Start MLflow run
mlflow.set_experiment("Insurance Premium Prediction")
with mlflow.start_run():
    # Fit model (if not already fitted)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_val)

    # Ensure no negative values for RMSLE
    preds = np.clip(preds, 0, None)
    y_val_clipped = np.clip(y_val, 0, None)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val_clipped, preds))
    mae = mean_absolute_error(y_val_clipped, preds)
    r2 = r2_score(y_val_clipped, preds)
    rmsle = np.sqrt(np.mean((np.log1p(preds) - np.log1p(y_val_clipped)) ** 2))

    # Log metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSLE", rmsle)

    # Log model with example input
    mlflow.sklearn.log_model(best_model, "model", input_example=X_log.iloc[:1])

    print(f"Logged model with RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}, RMSLE={rmsle:.4f}")
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Premium Predictor")
st.write("Enter customer details to predict the premium amount.")

# --- User Input ---
def user_input():
    data = {
        "Age": st.number_input("Age", 18, 100, 35),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Annual Income": st.number_input("Annual Income", 0, 1_000_000, 50000),
        "Marital Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        "Number of Dependents": st.number_input("Dependents", 0, 10, 1),
        "Education Level": st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"]),
        "Occupation": st.selectbox("Occupation", ["Salaried", "Self-Employed", "Retired"]),
        "Health Score": st.slider("Health Score", 0, 100, 50),
        "Location": st.selectbox("Location", ["Urban", "Suburban", "Rural"]),
        "Previous Claims": st.number_input("Previous Claims", 0, 10, 0),
        "Vehicle Age": st.number_input("Vehicle Age", 0, 20, 5),
        "Credit Score": st.number_input("Credit Score", 300, 900, 650),
        "Insurance Duration": st.number_input("Insurance Duration", 0, 10, 3),
        "Policy Start Date": st.date_input("Policy Start Date"),
        "Customer Feedback": st.selectbox("Feedback", ["Poor", "Average", "Good"]),
        "Smoking Status": st.selectbox("Smoking Status", ["Yes", "No"]),
        "Exercise Frequency": st.selectbox("Exercise Frequency", ["Never", "Rarely", "Monthly", "Weekly", "Daily"]),
        "Property Type": st.selectbox("Property Type", ["House", "Apartment", "Condo"]),
        "Policy Type": st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    }
    return pd.DataFrame([data])

# --- Preprocessing ---
def preprocess(df):
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    
    # Calculate missing columns expected by the model
    df['Policy Age (Days)'] = (pd.Timestamp.today() - df['Policy Start Date']).dt.days
    df['Policy Start Year'] = df['Policy Start Date'].dt.year
    df['Policy Start Month'] = df['Policy Start Date'].dt.month
    df['Policy Start Day'] = df['Policy Start Date'].dt.day

    # Interaction features and risk flags (optional, only if model was trained with them)
    df['Age_x_Health'] = df['Age'] * df['Health Score']
    df['Credit_x_Claims'] = df['Credit Score'] * df['Previous Claims']
    df['Income_x_Credit'] = df['Annual Income'] * df['Credit Score']
    df['Is_Smoker'] = df['Smoking Status'].str.lower().map({'yes':1,'no':0})
    df['Low_Credit_Score'] = (df['Credit Score'] < 600).astype(int)
    df['Multiple_Claims'] = (df['Previous Claims'] > 2).astype(int)
    df['Customer_Feedback_Score'] = df['Customer Feedback'].map({'Poor':0,'Average':1,'Good':2})
    df['Exercise_Freq_Score'] = df['Exercise Frequency'].map({'Never':0,'Rarely':1,'Monthly':2,'Weekly':3,'Daily':4})

    # Convert categorical columns to string (matching model)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    return df

# --- Prediction ---
input_df = user_input()
if st.button("Estimate Premium"):
    processed = preprocess(input_df)
    pred_log = model.predict(processed)
    pred = np.expm1(pred_log)  # Reverse log-transform
    st.success(f"Estimated Premium: ₹{pred[0]:,.2f}")

import pickle
import os
# Save the best model
model_file = os.path.join(model_path, "best_model.pkl")
with open(model_file, "wb") as f:
    pickle.dump(best_model, f)

