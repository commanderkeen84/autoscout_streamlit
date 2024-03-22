# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:51:46 2024

@author: reinh
"""

import streamlit as st 
import pandas as pd # For creating data frame
import matplotlib.pyplot as plt # For plotting chart


# Creating data frame by reading csv file
df = pd.read_csv('autoscout24.csv') 

# remove null values duplicates  
df = df.dropna(axis=0)
df.drop_duplicates(inplace=True)
    
    
def app():
    st.title("[Autoscout Daten](https://www.kaggle.com/datasets/ander289386/cars-germany/code)")  # Erzeugt eine Titel für die App

    #st.info("Wähle eine Seite im Navi links.")    # Erzeugt eine Infobox

    st.info("""
    Diese Homepage präsentiert die Datenanalyse des Automarkts anhand von Verkaufsdaten von Autoscout24. Autor: Reinhard Uehleke
    """)

    st.header("Datensatz")
    st.write(df.head())

    st.subheader("Remove null values and duplicates")  
    st.code('''  
    df = df.dropna(axis=0)
    df.drop_duplicates(inplace=True)
    ''')
  
    # Group by year and count the number of rows per year
    rows_per_year = df.groupby('year').size()


    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(rows_per_year.index, rows_per_year.values, color='skyblue')
    ax.set_xlabel('Jahr', fontsize=14)
    ax.set_ylabel('Anzahl', fontsize=14)
    ax.set_title('Verkäufe nach Jahr', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    st.write("""
    Durchschnittle Anzahl verkaufter Autos pro Jahr: 4188
    """)
    
    #st.header("Verkäufe nach Antriebsart")
    # Count the occurrences of each fuel type
    fuel_counts = df['fuel'].value_counts()
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    fuel_counts.plot(kind='bar', color='skyblue', ax=ax)
    plt.title('Verkäufe nach Antriebsart', fontsize=16)
    plt.xlabel('Antriebsart', fontsize=14)
    plt.ylabel('Anzahl', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)
            