@echo off
REM Simple launcher: ouvre deux fenêtres PowerShell (Back et Front)
REM Backend: cd Back ; Activate.ps1 ; python -m uvicorn api:app --reload
REM Frontend: cd Front ; npm run dev

SETROOT=%~dp0

REM Backend window: use -File to avoid complex inline quoting issues
start powershell -NoExit -File "%~dp0launch_back.ps1"

REM Frontend window
start powershell -NoExit -File "%~dp0launch_front.ps1"

exit /b 0
