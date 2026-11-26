@echo off
REM Batch-compile portal_ml_5chambers.vmf for Portal 1 (Source 2007/2013 SP)
REM Override BIN_DIR and GAME_DIR if installed outside default Steam path.

if "%BIN_DIR%"=="" set "BIN_DIR=C:\Program Files (x86)\Steam\steamapps\common\Portal\bin"
if "%GAME_DIR%"=="" set "GAME_DIR=C:\Program Files (x86)\Steam\steamapps\common\Portal\portal"

set "MAP_VMF=%~dp0portal_ml_5chambers.vmf"
set "MAP_BSP=%~dp0portal_ml_5chambers.bsp"

echo [1/3] Running vbsp...
"%BIN_DIR%\vbsp.exe" -game "%GAME_DIR%" "%MAP_VMF%" || exit /b 1

echo [2/3] Running vvis (fast)...
"%BIN_DIR%\vvis.exe" -fast -game "%GAME_DIR%" "%MAP_BSP%" || exit /b 1

echo [3/3] Running vrad (fast)...
"%BIN_DIR%\vrad.exe" -fast -game "%GAME_DIR%" "%MAP_BSP%" || exit /b 1

echo Copying compiled BSP to %GAME_DIR%\maps\...
copy /Y "%MAP_BSP%" "%GAME_DIR%\maps\portal_ml_5chambers.bsp"
echo Done!
