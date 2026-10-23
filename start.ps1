$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try { & python workbench.py; exit $LASTEXITCODE }
finally { Pop-Location }
