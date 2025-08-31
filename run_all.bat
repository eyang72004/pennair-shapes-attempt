@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv" ( python -m venv .venv )
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if not exist outputs mkdir outputs
set DYN=data\PennAir_2024_App_Dynamic.mp4
set HARD=data\PennAir_2024_App_Dynamic_Hard.mp4
set DUMMY=data\dummy_input.mp4
if not exist "%DYN%" set DYN=%DUMMY%
if not exist "%HARD%" set HARD=%DUMMY%
python src\run_static.py --in data\PennAir_2024_App_Static.png --out outputs\static_annotated.png --annotate_xyz
python src\run_video.py --in "%DYN%" --out outputs\dynamic_annotated.mp4 --annotate_xyz --draw_trails --also_avi
python src\run_video.py --in "%HARD%" --out outputs\dynamic_hard_annotated.mp4 --annotate_xyz --draw_trails --also_avi
echo [done] Check outputs\ for results.
endlocal
