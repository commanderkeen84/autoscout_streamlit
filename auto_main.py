import streamlit as st 
from matplotlib import widgets
# delete the codespace if new modules are not working
# then redeploy the codespace  

# Importiere die Seiten/Module
import Datensatz
import Datensatz_start
import ML 
import hyper 
import evalu 
import prediction 

# use codespaces in streamlit to edit and see changes 

# ------------------------------------------
# -----------------Preamble-----------------
# ------------------------------------------

pages = {
    "1. Datensatz": Datensatz_start,
    "2. Daten": Datensatz,
    "3. Compare ML-Models": ML,
    "4. Hyperparameter tuning": hyper,
    "5. Evaluation des ML-Models": evalu,
    "6. Car price prediction": prediction
}

st.sidebar.title("Seiten")
select = st.sidebar.radio("Gehe zu Seite:", list(pages.keys()))
pages[select].app()   # Startet die Seite
