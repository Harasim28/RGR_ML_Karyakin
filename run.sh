#!/bin/bash
# Скрипт запуска дашборда

cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
