@echo off
timeout /t 30 /nobreak
cd "C:\Users\Arya DS\Documents\pitong"

set TF_ENABLE_ONEDNN_OPTS=0
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8

"C:\Users\Arya DS\Documents\pitong\venv311\Scripts\python.exe" main.py
pause