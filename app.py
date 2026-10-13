"""Entry point: sidebar navigation.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Term-Deposit Predictor", page_icon="🏦", layout="wide")
from src.dashboard import pages as ui  # noqa: E402

st.markdown(ui.CSS, unsafe_allow_html=True)
pg = st.navigation({
    "Dashboard": [st.Page(ui.overview, title="Overview", icon="🏠", default=True),
                  st.Page(ui.explorer, title="Data Explorer", icon="📈"),
                  st.Page(ui.performance, title="Model Performance", icon="📊")],
    "Predict": [st.Page(ui.predict, title="Predict a Client", icon="🔮"),
                st.Page(ui.batch, title="Batch Prediction", icon="📂")],
    "Reasoning": [st.Page(ui.logic, title="Logic & Rules", icon="🧠"),
                  st.Page(ui.about, title="About", icon="ℹ️")]})
with st.sidebar:
    st.divider()
    st.markdown("**🎯 Target `y`**  \nDid the client subscribe to a term deposit? (1 = yes, 0 = no)")
    st.caption("Pre-call information only: call duration is excluded because it is not known before the call.")
pg.run()
