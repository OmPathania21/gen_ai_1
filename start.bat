@echo off
rem Start VERITAS at http://localhost:8000 (Windows). First run: py scripts\setup.py
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Setup has not been run yet. Run:  py scripts\setup.py
  exit /b 1
)
findstr /r /c:"^GROQ_API_KEY=gsk_" .env >nul 2>&1 || echo Warning: no Groq API key in .env. Questions will fail until you add one ^(see README, step 3^).
findstr /c:"gsk_your_key_here" .env >nul 2>&1 && echo Warning: .env still has the placeholder key. Replace it with your own ^(see README, step 3^).

where npm >nul 2>&1
if %errorlevel%==0 (
  pushd web
  if not exist node_modules call npm install
  call npm run build
  popd
) else (
  if not exist "web\dist\index.html" echo Warning: Node.js is not installed, so the web page cannot be built ^(API only^).
)

echo VERITAS is starting at http://localhost:8000 (press Ctrl+C to stop)
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
