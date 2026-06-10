$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    & texify --pdf --quiet --synctex=1 --max-iterations=5 Doku.tex
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
