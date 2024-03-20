import streamlit as st 
from matplotlib import widgets

# Importiere die Seiten/Module
import Datensatz
import Datensatz_start
import ML 
import evalu 

# ------------------------------------------
# -----------------Preamble-----------------
# ------------------------------------------

pages = {
    "1. Datensatz": Datensatz_start,
    "2. Daten": Datensatz,
    "3. Machne learning": ML,
    "4. Evaluation des Models": evalu
}

st.sidebar.title("Seiten")
select = st.sidebar.radio("Gehe zu Seite:", list(pages.keys()))
pages[select].app()   # Startet die Seite
