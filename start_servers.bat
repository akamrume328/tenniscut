@echo off
echo Starting Flask Web Server and Celery Worker...

REM Activate virtual environment
call venv\Scripts\activate

REM Start Celery Worker in a new window
start "Celery Worker" cmd /k "celery -A tasks.celery worker --loglevel=info -P gevent"

REM Start Flask App in the current window
echo Starting Flask server...
python app.py

pause