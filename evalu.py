# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 01:06:15 2024

@author: reinh
"""

import streamlit as st 
import numpy as np # For linear algebra
import pandas as pd # For creating data frame
import matplotlib.pyplot as plt # For plotting chart
import seaborn as sns # For creating data visualization

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
   



def app():
    
    st.header("Machine learning Evaluation")
    
    #read DataFrame from pickle file
    df_5= pd.read_pickle("ML_data.pkl")
    
    # preparting the dataset 
    X = df_5.drop(columns='price')
    y = df_5['price']
    
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
    models = [
        ('Random Forest',     RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=8956165))
    
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
    
        # Extract feature names after one-hot encoding
        feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(input_features=categorical_features)
    
        # Concatenate feature names and numerical features
        all_feature_names = np.concatenate([feature_names, numerical_features])
    
        # Extract feature importances
        feature_importances = model.feature_importances_
    
        # Sort feature importances and names in descending order
        sorted_indices = feature_importances.argsort()[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = all_feature_names[sorted_indices]
    
        # Display top ten important features and their names in Streamlit
        st.subheader(f"Feature importance (top 10) of {name}")
        for i in range(10):
            st.markdown(f"**{i + 1}:** {sorted_feature_names[i]}: {sorted_importances[i]:.4f}")
    

        st.subheader(f"Predicted vs. actual proce of {name}")
        # Plot predicted price vs. actual price
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, color='blue', label='Predicted')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True Values')
        ax.set_xlabel('True Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted Price vs. Actual Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
    st.subheader("Further steps") 
    st.write("""
    - Improve accuracy by feature engineering, e.g., putting the models into classes of models to reduce nodes
    - Consider outlier exclusion more systematically
    - Improve the training with grid-/random search
    - Improve understanding of prediciton accuracy using conformized prediction intervals
    """)
                 
    st.subheader("Data limitatons")         
    st.write("""
    - Das Alter des Autos zum Zeitpunkt des Verkaufs ist unbekannt - nur das Jahr der Erstzulassung
    - Das Verkaufsjahr ist unbekannt - oder sind alle Verkäufe aus dem letzten Jahr (2021)? Beschreibung unklar. 
    - Es gibt keine Information zu Austattungsmerkmalen
    """)
                
                
                
                
                