import os
import streamlit as st
def pull_last_changes_from_remote_repo():
    os.system("git pull")
    os.system("git submodule update --init --recursive")
    #rerun streamlit from the beginning
    st.rerun()
    return