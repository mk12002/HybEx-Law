@echo off
REM ============================================================================
REM HybEx-Law System Interactive Runner
REM ============================================================================
REM This script provides a menu-driven interface to run various components
REM of the HybEx-Law system. It handles environment setup, execution, and
REM cleanup automatically.
REM ============================================================================

ECHO [INFO] Setting up the Python virtual environment...

REM Create a virtual environment named 'hybex-env'
python -m venv hybex-env
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to create the virtual environment. Make sure Python and venv are installed.
    GOTO :error
)

REM Activate the virtual environment
CALL hybex-env\Scripts\activate.bat
ECHO [INFO] Virtual environment activated.

ECHO [INFO] Installing required dependencies from requirements.txt...

REM Install dependencies
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to install dependencies. Check your requirements.txt file.
    GOTO :cleanup_and_error
)

ECHO [INFO] Dependencies installed successfully.

:MENU
cls
ECHO.
ECHO ============================================================================
ECHO HybEx-Law System Control Panel
ECHO ============================================================================
ECHO Please select an option:
ECHO.
ECHO   1 - Train System (using ./data/ directory)
ECHO   2 - Evaluate System
ECHO   3 - Predict Single Query
ECHO   4 - Preprocess Data (using ./data/ directory)
ECHO   5 - Check System Status
ECHO   6 - Start Interactive Chat Session
ECHO   7 - Show Help
ECHO.
ECHO   8 - Exit and Clean Up Environment
ECHO.
ECHO ============================================================================

set /p "CHOICE=Enter your choice [1-8]: "

IF "%CHOICE%"=="1" GOTO TRAIN
IF "%CHOICE%"=="2" GOTO EVALUATE
IF "%CHOICE%"=="3" GOTO PREDICT
IF "%CHOICE%"=="4" GOTO PREPROCESS
IF "%CHOICE%"=="5" GOTO STATUS
IF "%CHOICE%"=="6" GOTO CHAT
IF "%CHOICE%"=="7" GOTO HELP
IF "%CHOICE%"=="8" GOTO CLEANUP
ECHO "%CHOICE%" is not a valid choice.
pause
GOTO MENU

:TRAIN
ECHO [INFO] Executing: Train System...
python -m hybex_system.main train --data-dir data/
pause
GOTO MENU

:EVALUATE
ECHO [INFO] Executing: Evaluate System...
python -m hybex_system.main evaluate
pause
GOTO MENU

:PREDICT
ECHO [INFO] Executing: Predict Single Query...
set /p "QUERY=Enter your legal query: "
python -m hybex_system.main predict --query "%QUERY%"
pause
GOTO MENU

:PREPROCESS
ECHO [INFO] Executing: Preprocess Data...
python -m hybex_system.main preprocess --data-dir data/
pause
GOTO MENU

:STATUS
ECHO [INFO] Executing: Check System Status...
python -m hybex_system.main status
pause
GOTO MENU

:CHAT
ECHO [INFO] Starting Interactive Chat Session...
python -m hybex_system.main chat
pause
GOTO MENU

:HELP
ECHO [INFO] Displaying Help Information...
python -m hybex_system.main --help
pause
GOTO MENU

:CLEANUP
ECHO [INFO] Cleaning up the environment...

REM Uninstall dependencies
ECHO [INFO] Uninstalling dependencies...
pip freeze > installed_packages.txt
pip uninstall -r installed_packages.txt -y > nul
del installed_packages.txt

REM Deactivate and clean up the virtual environment
ECHO [INFO] Deactivating and removing the virtual environment...
deactivate
rmdir /s /q hybex-env

ECHO [SUCCESS] Process completed and environment cleaned up.
GOTO :end

:cleanup_and_error
ECHO [CLEANUP] An error occurred. Cleaning up before exiting...
IF EXIST "hybex-env\Scripts\activate.bat" (
    pip freeze > installed_packages.txt
    pip uninstall -r installed_packages.txt -y > nul
    del installed_packages.txt
    deactivate
    rmdir /s /q hybex-env
)
ECHO [CLEANUP] Cleanup complete.
GOTO :error

:error
ECHO [FAILURE] The script exited with an error.
pause
exit /b 1

:end
ECHO [SUCCESS] Script finished.
exit /b 0
