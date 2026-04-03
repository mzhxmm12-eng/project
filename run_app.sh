#!/bin/bash
cd "$(dirname "$0")"
export DEEPSEEK_API_KEY="your_key_here"

# Start FastAPI chat backend on port 8502
python -m uvicorn app.chat_api:app --port 8502 --host 0.0.0.0 &
FASTAPI_PID=$!

# Start Streamlit on port 8501
python -m streamlit run app/streamlit_app.py --server.port 8501

# Cleanup on exit
kill $FASTAPI_PID
