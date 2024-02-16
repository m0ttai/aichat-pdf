#! /bin/bash

### pip for python3.11 install ###
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

### Frontend framework ###
python -m pip install streamlit streamlit_chat qdrant_client langchain-community openai bs4 PyPDF2
