param([string]$Log = "logs\strength_chain_final.log")

$PY = "C:\Research work\freqlite\.venv\Scripts\python.exe"
$Root = "C:\Research work\freqlite"
Set-Location $Root

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Run($desc, [string[]]$args_) {
    Log "=== $desc ==="
    & $PY @args_ 2>&1 | Tee-Object -Append -FilePath $Log
    if ($LASTEXITCODE -ne 0) { Log "WARNING: exit code $LASTEXITCODE for $desc" }
}

Log "RESUME_CHAIN_START"

Run "STAGE C: L96 iT+TN on ETTm1,ETTm2,weather" `
    "scripts/run_experiments.py","--models","itransformer,timesnet","--lookback","96","--datasets","ETTm1,ETTm2,weather"

Run "STAGE D: ECL iT+TN L336" `
    "scripts/run_experiments.py","--datasets","electricity","--models","itransformer,timesnet","--lookback","336"

Run "STAGE E: Traffic full grid" `
    "scripts/run_experiments.py","--models","naive,nlinear,dlinear,rlinear,fits,freqlite,itransformer,timesnet","--datasets","traffic"

Run "STAGE F: make_shift_table"  "scripts/make_shift_table.py"
Run "STAGE F: make_tables"       "scripts/make_tables.py"
Run "STAGE F: make_figures"      "scripts/make_figures.py"
Run "STAGE F: significance"      "scripts/significance.py"

Log "STRENGTH_CHAIN_DONE"
