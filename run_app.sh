#!/bin/bash
cd "$(dirname "$0")"
streamlit run app/streamlit_app.py --server.port 8501
