@echo off
setlocal
cd /d "%~dp0"
if /I "%~1"=="--help" goto help
where python >nul 2>nul
if errorlevel 1 (
 echo Python was not found. Activate your existing Conda base or ml environment first.
 exit /b 1
)
python -B -c "import importlib.util,sys; missing=[n for n in ['streamlit', 'pandas', 'numpy', 'scipy', 'sklearn', 'nltk'] if importlib.util.find_spec(n) is None]; print('Missing dependencies: '+', '.join(missing) if missing else 'Search Lab: dependencies found'); sys.exit(bool(missing))"
if errorlevel 1 (
 echo Use an environment with this project's requirements. This launcher installs nothing.
 exit /b 1
)
if /I "%~1"=="--check" exit /b 0
python -m streamlit run app.py --server.address 127.0.0.1 %*
exit /b %errorlevel%
:help
echo Search Lab
echo run.bat --check     Check dependencies without opening the app.
echo run.bat             Start the local browser app.
echo run.bat --server.port 8502     Choose another local port.
exit /b 0
