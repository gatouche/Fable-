@echo off
REM Enchaine les passages du detecteur de double rejet et consigne tout
REM dans resultats_rejet.txt (a renvoyer tel quel).
REM
REM Si "py" ne marche pas, remplace-le par le chemin complet de python.exe
REM a la ligne SET PY ci-dessous.

setlocal
set PY=py
set F=ticks_ES_full.csv
set LOG=resultats_rejet.txt

if not exist "%F%" (
    echo Fichier introuvable : %F%
    echo Corrige la ligne "set F=" dans ce .bat
    pause
    exit /b 1
)

if exist "%LOG%" del "%LOG%"

call :passage "1. Reference - confirm 30, risque 8"          ""
call :passage "2. Ton setup reel - stop serre"               "--risque 2"
call :passage "3. Confirmation rapide - 10s"                 "--confirm 10 --risque 3"
call :passage "4. Confirmation tres rapide - 5s"             "--confirm 5 --risque 3"
call :passage "5. Stop temporel - horizon 5 min"             "--risque 2 --horizon 5"
call :passage "6. Sweeps plus francs - 3 niveaux mini"       "--niveaux 3 --risque 3"

echo.
echo ==========================================================
echo Termine. Envoie le fichier %LOG%
echo ==========================================================
pause
exit /b 0


:passage
echo.
echo ==========================================================
echo %~1
echo ==========================================================
echo. >> "%LOG%"
echo ########################################################## >> "%LOG%"
echo # %~1 >> "%LOG%"
echo # commande : rejet_sequence.py %F% %~2 >> "%LOG%"
echo ########################################################## >> "%LOG%"
%PY% rejet_sequence.py %F% %~2 >> "%LOG%" 2>&1
type nul
exit /b 0
