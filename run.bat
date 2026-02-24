@echo off
cd /d "%~dp0"

echo Creating uv environment...
uv venv
if errorlevel 1 exit /b 1

echo Syncing dependencies...
uv sync
if errorlevel 1 exit /b 1

echo Training models (if not already present)...
uv run python train_models.py
if errorlevel 1 exit /b 1

echo Starting Streamlit app...
uv run streamlit run app.py
