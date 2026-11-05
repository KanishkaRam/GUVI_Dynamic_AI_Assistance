import json
import time
import pandas as pd
import streamlit as st
from config.settings import EMBEDDING_MODELS
from src.pipeline import active_artifacts, build_knowledge_base

def artifacts():
    try:
        return active_artifacts()
    except (FileNotFoundError, ValueError) as error:
        st.info(str(error))
        st.stop()


