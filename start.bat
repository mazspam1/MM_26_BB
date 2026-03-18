@echo off
setlocal EnableDelayedExpansion
title CBB Lines - NCAAB Prediction System
color 0A
cls

:MENU
echo.
echo  ============================================================
echo    CBB Lines - NCAAB Prediction System  (PhD-Grade v2.0)
echo  ============================================================
echo.
echo    MAIN
echo    ----
echo    [1]  Full Daily Run      - ingest + ratings + predict + dashboard
echo    [2]  Quick Predict        - predictions only + dashboard (needs data)
echo    [3]  PhD Pipeline         - full pipeline with Bayesian/Injuries/RAPM
echo.
echo    INDIVIDUAL STEPS
echo    ----------------
echo    [4]  Ingest Data          - fetch schedule, box scores, odds
echo    [5]  Calculate Ratings    - KenPom-style adjusted efficiency
echo    [6]  Generate Predictions - spread/total predictions for today
echo    [7]  Fetch Splits         - DraftKings betting splits
echo.
echo    SERVICES
echo    --------
echo    [8]  Start Dashboard      - web dashboard only (port 2501)
echo    [9]  Start API            - FastAPI server only (port 2500)
echo    [10] Start Worker         - background scheduler
echo    [11] Stop All Services    - kill everything
echo    [12] Check Status         - see what's running
echo.
echo    ANALYSIS
echo    --------
echo    [13] Run Backtest         - 30-day rolling backtest
echo    [14] Fit Calibration      - calibrate model from backtest
echo    [15] Quality Report       - data quality report
echo.
echo    TOURNAMENT
echo    -----------
echo    [16] Tournament Sim       - March Madness simulation
echo    [17] Render Bracket       - visual bracket HTML
echo.
echo    OTHER
echo    -----
echo    [18] Run Tests            - pytest test suite
echo    [19] Setup/Install        - install dependencies
echo    [20] Help                 - all commands
echo    [Q]  Quit
echo.
echo  ============================================================
set /p CHOICE="  Select option [1-20 or Q]: "

if "%CHOICE%"=="1"  goto FULL
if "%CHOICE%"=="2"  goto QUICK
if "%CHOICE%"=="3"  goto PIPELINE
if "%CHOICE%"=="4"  goto INGEST
if "%CHOICE%"=="5"  goto RATINGS
if "%CHOICE%"=="6"  goto PREDICT
if "%CHOICE%"=="7"  goto SPLITS
if "%CHOICE%"=="8"  goto DASHBOARD
if "%CHOICE%"=="9"  goto API
if "%CHOICE%"=="10" goto WORKER
if "%CHOICE%"=="11" goto STOP
if "%CHOICE%"=="12" goto STATUS
if "%CHOICE%"=="13" goto BACKTEST
if "%CHOICE%"=="14" goto CALIBRATE
if "%CHOICE%"=="15" goto REPORT
if "%CHOICE%"=="16" goto TOURNAMENT
if "%CHOICE%"=="17" goto BRACKET
if "%CHOICE%"=="18" goto TESTS
if "%CHOICE%"=="19" goto SETUP
if "%CHOICE%"=="20" goto HELP
if /i "%CHOICE%"=="Q" goto QUIT

echo.
echo  Invalid choice. Try again.
timeout /t 2 >nul
cls
goto MENU

:FULL
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" full
goto DONE

:QUICK
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" quick
goto DONE

:PIPELINE
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" pipeline
goto DONE

:INGEST
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" ingest
goto DONE

:RATINGS
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" ratings
goto DONE

:PREDICT
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" predict
goto DONE

:SPLITS
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" splits
goto DONE

:DASHBOARD
cls
echo  Starting Dashboard on port 2501...
echo  Open: http://localhost:2501
echo.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" dashboard
goto DONE

:API
cls
echo  Starting API on port 2500...
echo  Open: http://localhost:2500/docs
echo.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" api
goto DONE

:WORKER
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" worker
goto DONE

:STOP
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" stop
echo.
echo  Press any key to return to menu...
pause >nul
cls
goto MENU

:STATUS
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" status
echo.
echo  Press any key to return to menu...
pause >nul
cls
goto MENU

:BACKTEST
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" backtest
goto DONE

:CALIBRATE
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" calibrate
goto DONE

:REPORT
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" report
goto DONE

:TOURNAMENT
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" tournament
goto DONE

:BRACKET
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" bracket
goto DONE

:TESTS
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" test
goto DONE

:SETUP
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" setup
goto DONE

:HELP
cls
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" help
goto DONE

:DONE
echo.
echo  ============================================================
echo    Done. Press any key to return to menu...
echo  ============================================================
pause >nul
cls
goto MENU

:QUIT
echo.
echo  Goodbye!
timeout /t 1 >nul
exit /b 0
