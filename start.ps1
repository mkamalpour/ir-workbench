$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Error 'Create .venv and install requirements.txt first. See README.md.'
    exit 1
}
Push-Location $PSScriptRoot
try { & $projectPython workbench.py; exit $LASTEXITCODE }
finally { Pop-Location }
