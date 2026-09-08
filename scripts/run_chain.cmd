@echo off
setlocal
cd /d "C:\Research work\freqlite"
set PYTHONUNBUFFERED=1
set PY=C:\Research work\freqlite\.venv\Scripts\python.exe
set LOG=logs\strength_chain_final.log

echo === CHAIN START %DATE% %TIME% === >> %LOG%

echo === STAGE C: L96 iT+TN ETTm1,ETTm2,weather === >> %LOG%
"%PY%" scripts\run_experiments.py --models itransformer,timesnet --lookback 96 --datasets ETTm1,ETTm2,weather >> %LOG% 2>&1

echo === STAGE D: ECL iT+TN L336 === >> %LOG%
"%PY%" scripts\run_experiments.py --datasets electricity --models itransformer,timesnet --lookback 336 >> %LOG% 2>&1

echo === STAGE E: Traffic full grid === >> %LOG%
"%PY%" scripts\run_experiments.py --models naive,nlinear,dlinear,rlinear,fits,freqlite,itransformer,timesnet --datasets traffic >> %LOG% 2>&1

echo === STAGE F: tables === >> %LOG%
"%PY%" scripts\make_shift_table.py >> %LOG% 2>&1
"%PY%" scripts\make_tables.py >> %LOG% 2>&1
"%PY%" scripts\make_figures.py >> %LOG% 2>&1
"%PY%" scripts\significance.py >> %LOG% 2>&1

echo === STRENGTH_CHAIN_DONE %DATE% %TIME% === >> %LOG%
