"""Run: python -m streamlit run app/main.py"""
import os
import sys

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
st.set_page_config(page_title='Dynamic AI Knowledge Assistant', page_icon='📚', layout='wide')
st.title('Dynamic AI Knowledge Assistant')
st.caption('Build a Wikipedia knowledge base, explore sources, and ask grounded questions.')
st.navigation([
    st.Page('pages/1_Dataset_Builder.py', title='Dataset Builder'),
    st.Page('pages/2_Knowledge_Base.py', title='Knowledge Base'),
    st.Page('pages/3_Chat_Assistant.py', title='Chat Assistant'),
    st.Page('pages/4_Analytics.py', title='Analytics'),
    st.Page('pages/5_Evaluation.py', title='Evaluation'),
]).run()
