# Cough AI - Run with uv
# Usage: .\run.ps1   (from cough_ai folder)

Set-Location $PSScriptRoot

Write-Host "Creating uv environment..." -ForegroundColor Cyan
uv venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Syncing dependencies..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training models (if not already present)..." -ForegroundColor Cyan
uv run python train_models.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting Streamlit app..." -ForegroundColor Green
uv run streamlit run app.py
