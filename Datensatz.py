# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:22:24 2024

@author: reinh
"""
import streamlit as st 
import pandas as pd # For creating data frame
import matplotlib.pyplot as plt # For plotting chart
import seaborn as sns # For creating data visualization

def app():
    
    st.header("Kennzahlen")
    
    # Creating data frame by reading csv file
    df = pd.read_csv('autoscout24.csv')


    # remove null values duplicates  
    df = df.dropna(axis=0)
    df.drop_duplicates(inplace=True)
    
    # davon Benzin Diesel Hybvrid Elektrisch gas 
    df['fuel'] = df['fuel'].replace({'CNG': 'CNG/LPG', 'LPG': 'CNG/LPG'})
    df['fuel'] = df['fuel'].replace({'Electric/Diesel': 'Others', 'Others ': 'CNG/LPG', '-/- (Fuel)': 'Others', 'Ethanol':'Others', 'Hydrogen':'Others' })
    
    # Pivot the DataFrame to create horizontal columns for each fuel and count the occurrences
    pivot_df = df.pivot_table(index='year', columns='fuel', aggfunc='size', fill_value=0)
    
    # Sum counts across different fuels for each year
    pivot_df['Total'] = pivot_df.sum(axis=1)
    
    # Calculate percentages for each fuel type relative to the total count of each year
    percentages = (pivot_df.drop('Total', axis=1) / pivot_df['Total'].values[:, None]) * 100
    
    # Convert percentages to integers and add '%' sign after each percentage value
    percentages = (percentages.round().astype(str) + '%').applymap(lambda x: x.ljust(len(x) + 5))
    
    # Concatenate percentages with the existing DataFrame
    pivot_df_with_percentage  = pd.concat([pivot_df, percentages], axis=1, keys=['Count', 'Percentage'])
    
    # Reorder the columns as desired
    column_order = [( 'Count', 'Total'), ('Count', 'Gasoline',), ('Percentage', 'Gasoline'), ('Count', 'Diesel'), ('Percentage', 'Diesel'), ('Count', 'Electric'), ('Percentage', 'Electric'), ('Count', 'Electric/Gasoline'), ('Percentage', 'Electric/Gasoline'), ('Count','Others'), ('Percentage', 'Others')]  # Change the order as needed
    pivot_df_with_percentage = pivot_df_with_percentage[column_order]
    
    # Print the resulting table
    print(pivot_df_with_percentage)
    
    
    # some outliers or coding errors seem present
    # 1. remove max value of milage above 600000
    df = df.drop(df[df['mileage'] >= 600000].index)
    
    # 2. remove prices above 600000
    df = df.drop(df[df['price'] >= 600000].index)
    
    
    # Filter the DataFrame to include only the top 15 values
    top_15_df = df['make'].value_counts().nlargest(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_15_df.index, y=top_15_df.values, palette='plasma', ax=ax)
    ax.set_title('Top 15 verkaufte Marken', fontsize=16)
    ax.set_xlabel('Marke', fontsize=12)
    ax.set_ylabel('Anzahl', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    st.subheader("1. Top verkaufte Marken")
    # Display the plot using st.pyplot()
    st.pyplot(fig)

    st.subheader("2. Verkaufspreise nach Marke")
    # Get the counts of each car make
    make_counts = df['make'].value_counts()
    
    # Create a boxplot sorted by counts
    fig, ax = plt.subplots(figsize=(15, 8))
    make_order = make_counts.index[:15]
    sns.boxplot(x='make', y='price', data=df, order=make_order, palette='pastel', ax=ax)
    ax.set_title('Preis nach Marke (Top 15)', fontsize=18)
    ax.set_xlabel('Marke', fontsize=14)
    ax.set_ylabel('Preis (in EUR)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    # Remove prices above 600000
    df = df.drop(df[df['price'] > 200000].index)
    
    st.subheader("3. Verkaufspreise nach Marke (unter €200000)")
    # Get the counts of each car make
    make_counts = df['make'].value_counts()
    
    # Create a boxplot sorted by counts
    fig, ax = plt.subplots(figsize=(15, 8))
    make_order = make_counts.index[:15]
    sns.boxplot(x='make', y='price', data=df, order=make_order, palette='pastel', ax=ax)
    ax.set_title('Preis nach Marke (Top 15))', fontsize=18)
    ax.set_xlabel('Marke', fontsize=14)
    ax.set_ylabel('Preis (in EUR)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    # save dataframe to use in next chapter ML 
    df.to_csv("my_data.csv")


    st.subheader("4. Preis nach Antrieb")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='fuel', y='price', data=df, palette='pastel', ax=ax)
    ax.set_title('Preis nach Antrieb)', fontsize=18)
    ax.set_xlabel('Antrieb', fontsize=14)
    ax.set_ylabel('Preis (in EUR)', fontsize=14)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    
    st.subheader("5. Kilometerstand nach Marke")
    # Get the counts of each car make
    make_counts = df['make'].value_counts()
    
    # Create a boxplot sorted by counts
    fig, ax = plt.subplots(figsize=(15, 8))
    make_order = make_counts.index[:15]
    sns.boxplot(x='make', y='mileage', data=df, order=make_order, palette='pastel', ax=ax)
    ax.set_title('Kilometerstand nach Marke (Top 15))', fontsize=18)
    ax.set_xlabel('Marke', fontsize=14)
    ax.set_ylabel('Kilometer', fontsize=14)
    plt.xticks(rotation=45)
    
    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    st.subheader("6. Preis nach Jahr")
    # Create a boxplot 
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='year', y='price', data=df, palette='pastel', ax=ax)
    ax.set_title('Pries nach Jahr)', fontsize=18)
    ax.set_xlabel('Jahr', fontsize=14)
    ax.set_ylabel('Kilometer', fontsize=14)
    plt.xticks(rotation=45)
    
    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    
    
    st.subheader("7. Preis nach Kilometerstand (Diesel und Benzin)")
    # Filter the DataFrame to include only "Diesel" and "Gasoline" categories
    df_filtered = df[df['fuel'].isin(['Diesel', 'Gasoline'])]
    
    # Create a scatter plot with regression lines for each fuel type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_filtered, x='mileage', y='price', hue='fuel', palette='coolwarm', s=100, ax=ax)
    sns.regplot(data=df_filtered[df_filtered['fuel'] == 'Gasoline'], x='mileage', y='price', scatter=False, color='orange', label='Gasoline', lowess=True, ax=ax)
    sns.regplot(data=df_filtered[df_filtered['fuel'] == 'Diesel'], x='mileage', y='price', scatter=False, color='blue', label='Diesel', lowess=True, ax=ax)
    ax.set_title('Preis vs. Kilometerstand', fontsize=16)
    ax.set_xlabel('Kilometerstand', fontsize=14)
    ax.set_ylabel('Preis', fontsize=14)
    ax.grid(True)
    
    # Pass the figure as an object to Streamlit
    st.pyplot(fig)
    

    # Filter the DataFrame to include only "Diesel" and "Gasoline" categories
    df_filtered = df[df['fuel'].isin(['Diesel', 'Gasoline'])]

    # Create a scatter plot with regression lines for each fuel type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_filtered, x='mileage', y='price', hue='fuel', palette='coolwarm', s=100, ax=ax)
    sns.regplot(data=df_filtered[df_filtered['fuel'] == 'Gasoline'], x='mileage', y='price', scatter=False, color='orange', label='Gasoline', lowess=True, ax=ax)
    sns.regplot(data=df_filtered[df_filtered['fuel'] == 'Diesel'], x='mileage', y='price', scatter=False, color='blue', label='Diesel', lowess=True, ax=ax)
    ax.set_title('Preis vs. Kilometerstand', fontsize=16)
    ax.set_xlabel('Kilometerstand', fontsize=14)
    ax.set_ylabel('Preis', fontsize=14)
    ax.grid(True)
    
    # Set y-axis limits
    ax.set_ylim(0, 20000)
    
    # Pass the figure as an object to Streamlit
    st.pyplot(fig)
 
