# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:51:46 2024

@author: reinh
"""


import streamlit as st 
import pandas as pd # For creating data frame
import matplotlib.pyplot as plt # For plotting chart
import seaborn as sns # For creating data visualization



def app():
    st.title("Car price prediction")  # Erzeugt eine Titel für die App

    #st.info("Wähle eine Seite im Navi links.")    # Erzeugt eine Infobox

    st.info("""
    Diese Homepage präsentiert die Datenanalyse des Automarkts anhand von Verkaufsdaten von Autoscout24 ([data-source](https://www.kaggle.com/datasets/ander289386/cars-germany/code)). 
    
    Autor: [Reinhard Uehleke](https://www.linkedin.com/in/reinhard-uehleke-9bb740262/) 
            
    [Code on Github](https://github.com/commanderkeen84/autoscout_streamlit)
    """)

    # Creating data frame by reading csv file
    df = pd.read_csv('autoscout24.csv') 

    st.header("Datensatz")
    st.write(df.head())

    st.subheader("1. Remove null values and duplicates")  
    st.code('''  
    df = df.dropna(axis=0)
    df.drop_duplicates(inplace=True)
    ''')
    
    # count, report and remove null values duplicates  
    duplicate_count = df.duplicated().sum()
    null_count = df.isnull().sum().sum()

    st.write(f"Gelöschte Beobachtungen:  \nAnzahl Duplikate: {duplicate_count}  \nAnzahl Missings: {null_count}")
    df = df.dropna(axis=0)
    df.drop_duplicates(inplace=True)

    st.subheader("2. Remove outliers")
    st.write("outliers present in price and mileage")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='mileage', y='price', data=df)
    plt.title('Scatterplot of Mileage vs Price')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.grid(True)
    st.pyplot(fig) 

    st.write("remove values of milage and price above 600.000")
    # some outliers or coding errors seem present
    # 1. remove max value of milage above 600000
    df = df.drop(df[df['mileage'] >= 600000].index)
    # 2. remove prices above 600000
    df = df.drop(df[df['price'] >= 600000].index)

    # save for next chapter 
    df.to_csv("df_clean.csv")


    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='mileage', y='price', data=df)
    plt.title('Scatterplot of Mileage vs Price')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.grid(True)
    st.pyplot(fig) 


    st.subheader("3. Number of sales per year")  
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
    Durchschnittliche Anzahl verkaufter Autos pro Jahr: ~4000  \n
             """)


    # Count the occurrences of each fuel type
    fuel_counts = df['fuel'].value_counts()
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fuel_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Verkäufe nach Antriebsart', fontsize=16)
    ax.set_xlabel('Antriebsart', fontsize=14)
    ax.set_ylabel('Anzahl', fontsize=14)
    plt.xticks(rotation=45)

    # Add counts on top of each bar
    for i, count in enumerate(fuel_counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom')
    
    # draw Graph 
    st.pyplot(fig)



            