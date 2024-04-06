# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:59:10 2024

@author: reinh
"""

import streamlit as st 
import numpy as np # For linear algebra
import pandas as pd # For creating data frame
import matplotlib.pyplot as plt # For plotting chart
import seaborn as sns # For creating data visualization

import json 

def app():
    
    st.header("Compare ML-Models")
    st.write("Focus on top 5 car makes to reduce computation time")

    #read DataFrame from pickle file
    df= pd.read_csv("mydata.csv")

    # Calculate counts for each make
    make_counts = df['make'].value_counts()
    
    # Group by 'make' and calculate the average price for each make
    average_price_by_make = df.groupby('make')['price'].mean()
    
    # Merge average prices with counts
    merged_df = pd.merge(average_price_by_make, make_counts, left_index=True, right_index=True)
    
    # Rename columns
    merged_df.columns = ['Average Price', 'Count']
    
    # Format average price to remove decimals
    merged_df['Average Price'] = merged_df['Average Price'].astype(int)
    
    # Rearrange the columns
    merged_df = merged_df[['Count', 'Average Price']]
    
    # Sort by counts and show only the top five
    sorted_df = merged_df.sort_values(by='Count', ascending=False).head(5)
    
    # Display the sorted DataFrame as a table in Streamlit
    st.subheader("Top 5 Marken")
    st.table(sorted_df)
    
    
    from sklearn.model_selection import train_test_split, cross_val_score 
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures,  StandardScaler
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    from sklearn.linear_model import LinearRegression, BayesianRidge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    
    from sklearn.metrics import  r2_score, mean_absolute_error, mean_squared_error, make_scorer
    import scipy.stats
    
    # Calculate counts for each make
    make_counts = df['make'].value_counts()
    
    # Get the top 5 most common makes
    df_5 = make_counts.nlargest(5).index.tolist()
    
    # Filter the original DataFrame for the top 5 makes
    df_5 = df[df['make'].isin(df_5)]
    

    st.subheader("Ausreißer nach Standardabweichungsmethode")
    st.code('''  
    mean = df_5['price'].mean()
    std_dev = df_5['price'].std()
    threshold = 3 * std_dev
    df_5.loc[df_5['price'].abs() > (mean + threshold), 'price'] = np.nan
    st.write(f"Anzahl Ausreißer: {df_5['price'].isnull().sum()}")
    df_5 = df_5.dropna(axis=0)
    ''')
    
    # Identify and remove outliers 
    mean = df_5['price'].mean()
    std_dev = df_5['price'].std()
    threshold = 3 * std_dev
    df_5.loc[df_5['price'].abs() > (mean + threshold), 'price'] = np.nan
    st.write(f"Anzahl Ausreißer: {df_5['price'].isnull().sum()}")
    df_5 = df_5.dropna(axis=0)
    
    # save dataframe to use in next chapter 
    df_5.to_csv("ML_data.csv")
  
    
    # preparting the dataset 
    X = df_5.drop(columns='price')
    y = df_5['price']
    
    st.subheader("Setup der Pipeline")
    st.code('''

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9874654)

# Define feature names
categorical_features = ['make', 'model', 'fuel', 'gear', 'offerType']
numerical_features = ['mileage', 'hp']  

# Define transformers
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Define preprocessor 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features), 
        ('num', numerical_transformer, numerical_features)
    ]
)

# Define models
models = [
    ('OLS', LinearRegression()),
    ('Polynomial regression (degree 2)', make_pipeline(PolynomialFeatures(degree=2), LinearRegression())),
    ('Random Forest Regressor (sklearn default hyperparamters, except max_features=0.3)', RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features=0.3, min_samples_split=2, n_estimators=100, random_state=8956165)),
    ('Boosted tree', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=8956165))
]

# Fit and evaluate each model
for name, model in models:
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Fit pipeline to training data
    pipeline.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = pipeline.predict(X_test)

    # Evaluate performance
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
        
    # use test accuracy in later script with json 
    if name == 'Random Forest Regressor (sklearn default hyperparameters, except max_features=0.3)':
        mae_rf_default = mae_test
        acc_scores = {'mae_rf_default': mae_rf_default}
        with open('mae_rf_default.json', 'w') as f:
            json.dump(acc_scores, f)

    # Make predictions on train data
    y_pred_train = pipeline.predict(X_train)
    
    # Evaluate performance
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
        
    # Display metrics in Streamlit
    st.write(f"#### {name}")
    st.write("test-accuracy:")
    st.write(f"""
        - Mean Squared Error (MSE): {round(mse_test, 1)}
        - Mean Absolute Error (MAE): {round(mae_test, 1)}
        """)
    st.write("train-accuracy:")
    st.write(f"""
        - Mean Squared Error (MSE): {round(mse_train, 1)}
        - Mean Absolute Error (MAE): {round(mae_train, 1)}
        """)        
    ''')
    
    
    st.subheader("Ergebnisse")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=9874654)
    
    # Define feature names
    categorical_features = ['make', 'model', 'fuel', 'gear', 'offerType']
    numerical_features = ['mileage', 'hp']  
    
    # Define transformers
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    # Define preprocessor 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features), 
            ('num', numerical_transformer, numerical_features)
        ]
    )

    # Define models
    st.write("RandomForestRegressor has the default setting of max_features=1, which corresponds to a bagged tree. Here use 0.3*n_features to determine max features at each split in each random forest. ")
    models = [
        ('OLS', LinearRegression()),
        ('Polynomial regression (degree 2)', make_pipeline(PolynomialFeatures(degree=2), LinearRegression())),
        ('Random Forest Regressor (sklearn default hyperparameters, except max_features=0.3)',  RandomForestRegressor(max_depth=None, min_samples_leaf=1, max_features=0.3, min_samples_split=2, n_estimators=100, random_state=8956165)),
        ('Boosted Random Forest Regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=8956165))
    ]
    
    # Fit and evaluate each model
    for name, model in models:
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit pipeline to training data
        pipeline.fit(X_train, y_train)
    
        # Make predictions on test data
        y_pred = pipeline.predict(X_test)
    
        # Evaluate performance
        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        
        # use test accuracy in later script with json 
        if name == 'Random Forest Regressor (sklearn default hyperparameters, except max_features=0.3)':
            mae_rf_default = mae_test
            acc_scores = {'mae_rf_default': mae_rf_default}
            with open('mae_rf_default.json', 'w') as f:
                json.dump(acc_scores, f)

        # Make predictions on train data
        y_pred_train = pipeline.predict(X_train)
    
        # Evaluate performance
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        
        # Display metrics in Streamlit
        st.write(f"#### {name}")
        st.write("test-accuracy:")
        st.write(f"""
            - Mean Squared Error (MSE): {round(mse_test, 1)}
            - Mean Absolute Error (MAE): {round(mae_test, 1)}
             """)
        st.write("train-accuracy:")
        st.write(f"""
            - Mean Squared Error (MSE): {round(mse_train, 1)}
            - Mean Absolute Error (MAE): {round(mae_train, 1)}
             """)
 
    st.subheader("Summary of comparing model performance")
    st.write("""
             - Random Forest erzielt leicht bessere (test-) accuracy als Poly
             - Train-accuracy >> test-accuracy for random forest indicates overfitting
             - Simple polynomial regression does not perform worse than the gradient boosted random forest   
             """)        