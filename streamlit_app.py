import streamlit as st

pg = st.navigation([st.Page("intro_page.py", title="Introduction to DIRTT"), st.Page("results_page.py",title="Results")])
pg.run()

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:75rem;
        }
    </style>
    """
)

