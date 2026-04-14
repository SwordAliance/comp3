@echo off
chcp 65001 >nul 2>&1
title COMP2 - Детекция дефектов авто
color 0A

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   COMP2 - Детекция дефектов авто     ║
echo  ╚══════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo  [ОШИБКА] Python не найден!
    echo.
    echo  Скачайте Python 3.10+ с https://python.org
    echo  При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add to PATH"
    echo.
    pause
    exit /b 1
)

:: First run — setup
if not exist venv (
    echo  [1/5] Первый запуск — создаю виртуальное окружение...
    python -m venv venv
    if errorlevel 1 (
        color 0C
        echo  [ОШИБКА] Не удалось создать venv
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

:: Check if packages installed
if not exist venv\Lib\site-packages\uvicorn (
    echo  [2/5] Устанавливаю PyTorch с CUDA 12.4 (это может занять 5-10 минут)...
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
    if errorlevel 1 (
        color 0C
        echo  [ОШИБКА] Не удалось установить PyTorch
        pause
        exit /b 1
    )

    echo  [3/5] Устанавливаю SAM2...
    pip install git+https://github.com/facebookresearch/sam2.git -q

    echo  [3/5] Устанавливаю остальные зависимости...
    pip install -r requirements.txt -q
    if errorlevel 1 (
        color 0C
        echo  [ОШИБКА] Не удалось установить зависимости
        pause
        exit /b 1
    )
)

:: Create directories
if not exist uploads mkdir uploads
if not exist results mkdir results
if not exist gallery mkdir gallery
if not exist checkpoints mkdir checkpoints

:: Download YOLO
if not exist yolo11m-seg.pt (
    echo  [4/5] Скачиваю модель YOLO (~44MB)...
    python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
)

:: Download SAM2 checkpoint
if not exist checkpoints\sam2.1_hiera_small.pt (
    echo  [5/5] Скачиваю модель SAM2 (~176MB)...
    python -c "import urllib.request, os; os.makedirs('checkpoints', exist_ok=True); urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', 'checkpoints/sam2.1_hiera_small.pt'); print('Done')"
)

:: Open firewall port 8000 (requires admin, silent fail if no rights)
netsh advfirewall firewall show rule name="COMP2" >nul 2>&1
if errorlevel 1 (
    echo  Открываю порт 8000 в фаерволе...
    netsh advfirewall firewall add rule name="COMP2" dir=in action=allow protocol=tcp localport=8000 >nul 2>&1
)

:: Get local IP for LAN access
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set LOCAL_IP=%%a
)
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo  ════════════════════════════════════════
echo   Всё готово! Запускаю сервер...
echo   Браузер откроется автоматически.
echo.
echo   На этом компе:  http://localhost:8000
echo   По сети:        http://%LOCAL_IP%:8000
echo.
echo   Чтобы остановить — просто закройте это окно.
echo  ════════════════════════════════════════
echo.

:: Open browser after 2 sec delay (in background)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

:: Start server (blocks until window closed)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
