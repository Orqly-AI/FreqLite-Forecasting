Set-Location 'C:\Research work\freqlite'
Write-Host '=== STAGE A: weather retry ===' 
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\run_experiments.py' --models itransformer,timesnet --datasets weather 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_a_weather_retry.log'
if ($LASTEXITCODE -ne 0) { Write-Host 'STAGE A failed, continuing anyway' }
Write-Host '=== STAGE B: shift stress ==='
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\run_shift_stress.py' 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_b_shift2.log'
if ($LASTEXITCODE -ne 0) { Write-Host 'STAGE B failed, continuing anyway' }
Write-Host '=== STAGE C: L96 new baselines ==='
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\run_experiments.py' --models itransformer,timesnet --lookback 96 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_c_L96.log'
if ($LASTEXITCODE -ne 0) { Write-Host 'STAGE C failed, continuing anyway' }
Write-Host '=== STAGE D: ECL new baselines ==='
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\run_experiments.py' --models itransformer,timesnet --datasets electricity 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_d_ecl.log'
if ($LASTEXITCODE -ne 0) { Write-Host 'STAGE D failed, continuing anyway' }
Write-Host '=== STAGE E: Traffic full grid ==='
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\run_experiments.py' --models naive,nlinear,dlinear,rlinear,fits,freqlite,itransformer,timesnet --datasets traffic 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_e_traffic.log'
if ($LASTEXITCODE -ne 0) { Write-Host 'STAGE E failed, continuing anyway' }
Write-Host '=== STAGE F: Tables / Figures / Significance ==='
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\make_shift_table.py' 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_f_shift_table.log'
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\make_tables.py' 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_f_tables.log'
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\make_figures.py' 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_f_figures.log'
& 'C:\Research work\freqlite\.venv\Scripts\python.exe' 'C:\Research work\freqlite\scripts\significance.py' 2>&1 | Tee-Object 'C:\Research work\freqlite\logs\strength_f_significance.log'
Write-Host '=== STRENGTH_DONE ==='
