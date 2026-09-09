$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Fermi Podcast Companion: Full Pipeline" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Transcribe, Chunk, and Index into Chroma
Write-Host "`n[1/3] Building the core index (Transcribe -> Chunk -> Embed)..." -ForegroundColor Yellow
python run_scale_up.py

# 2. Generate Episode Summaries (for recommendations)
Write-Host "`n[2/3] Generating Episode Summaries..." -ForegroundColor Yellow
python generate_summaries.py

# 3. Launch the Streamlit App
Write-Host "`n[3/3] Launching the interactive Streamlit UI..." -ForegroundColor Yellow
python -m streamlit run app.py
