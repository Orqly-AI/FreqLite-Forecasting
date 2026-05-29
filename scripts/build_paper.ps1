<#
.SYNOPSIS
    Build the FreqLite LaTeX manuscript to PDF using TinyTeX (pdflatex + bibtex).

.DESCRIPTION
    Runs the standard latex -> bibtex -> latex -> latex sequence inside paper/,
    using the TinyTeX installation in %APPDATA%\TinyTeX. Reports the resulting
    page count and scans the log for errors / undefined references / citations.

.PARAMETER Clean
    Remove LaTeX auxiliary files (.aux .bbl .blg .log .out .toc) before building.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_paper.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_paper.ps1 -Clean
#>
[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# --- Locate paper directory (script lives in scripts/, paper/ is sibling) ---
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$PaperDir  = Join-Path $RepoRoot 'paper'
$MainName  = 'main'
if (-not (Test-Path (Join-Path $PaperDir "$MainName.tex"))) {
    throw "Cannot find $MainName.tex in $PaperDir"
}

# --- Locate TinyTeX binaries (add to PATH for this process) ---
$TinyBin = Join-Path $env:APPDATA 'TinyTeX\bin\windows'
if (Test-Path $TinyBin) {
    $env:PATH = "$TinyBin;$env:PATH"
}
$pdflatex = (Get-Command pdflatex -ErrorAction SilentlyContinue)
$bibtex   = (Get-Command bibtex   -ErrorAction SilentlyContinue)
if (-not $pdflatex) { throw "pdflatex not found. Is TinyTeX installed at $TinyBin ?" }
if (-not $bibtex)   { throw "bibtex not found. Is TinyTeX installed at $TinyBin ?" }

Push-Location $PaperDir
try {
    if ($Clean) {
        Write-Host "Cleaning auxiliary files..." -ForegroundColor Cyan
        Get-ChildItem -File -Include *.aux,*.bbl,*.blg,*.log,*.out,*.toc,*.lof,*.lot,*.spl,*.fdb_latexmk,*.fls `
            -Path . -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }

    function Invoke-Step([string]$Exe, [string[]]$ArgList, [string]$Label) {
        Write-Host ">> $Label" -ForegroundColor Cyan
        & $Exe @ArgList
        # pdflatex returns nonzero on errors; bibtex returns nonzero on warnings.
        # We do not hard-fail mid-sequence (later passes resolve refs); the final
        # log scan below is the authoritative error check.
    }

    $latexArgs = @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', "$MainName.tex")

    Invoke-Step 'pdflatex' $latexArgs 'pdflatex (pass 1/3)'
    Invoke-Step 'bibtex'   @($MainName) 'bibtex'
    Invoke-Step 'pdflatex' $latexArgs 'pdflatex (pass 2/3)'
    Invoke-Step 'pdflatex' $latexArgs 'pdflatex (pass 3/3)'

    # --- Report ---
    $pdf = Join-Path $PaperDir "$MainName.pdf"
    if (-not (Test-Path $pdf)) {
        throw "Build failed: $MainName.pdf was not produced. See $MainName.log."
    }

    Write-Host ""
    Write-Host "PDF built: $pdf" -ForegroundColor Green

    # Page count from the log (pdflatex prints e.g. 'Output written on main.pdf (7 pages').
    $log = Get-Content "$MainName.log" -Raw
    $pageMatch = [regex]::Match($log, "Output written on .*\((\d+)\s+page")
    if ($pageMatch.Success) {
        Write-Host ("Pages: {0}" -f $pageMatch.Groups[1].Value) -ForegroundColor Green
    }

    # Scan for problems.
    $logLines = Get-Content "$MainName.log"
    $errors   = $logLines | Select-String -Pattern '^(.*:\d+:|!) '   # file:line: or '! ' TeX errors
    $undefRef = $logLines | Select-String -Pattern 'There were undefined references|Citation .* undefined|Reference .* undefined|LaTeX Warning: There were'

    if ($errors) {
        Write-Host "`nLaTeX ERRORS detected:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    } else {
        Write-Host "No LaTeX errors detected." -ForegroundColor Green
    }

    if ($undefRef) {
        Write-Host "`nUndefined reference/citation WARNINGS:" -ForegroundColor Yellow
        $undefRef | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    } else {
        Write-Host "No undefined references/citations." -ForegroundColor Green
    }

    if ($errors) { exit 1 }
}
finally {
    Pop-Location
}
