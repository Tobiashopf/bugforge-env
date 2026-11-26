param([string]$Python, [switch]$ControlsOnly, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $candidate = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $candidate) { $Python = $candidate }
    if (-not $Python) {$Python = (Get-Command python -CommandType Application | Select-Object -First 1).Source}
}
$old = $env:PYTHONPATH
$parent = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'BugForge-Diagnostic'
if ($OutputDirectory) { $parent = [IO.Path]::GetFullPath($OutputDirectory) }
$session = Join-Path $parent ([Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($session) | Out-Null
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $driverArgs = @('--out',$session)
    if ($ControlsOnly) { $driverArgs += '--controls-only' }
    & $Python -X utf8 (Join-Path $PSScriptRoot 'diagnostic.py') @driverArgs
    $code=$LASTEXITCODE
} finally {
    $env:PYTHONPATH=$old
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($session,($session+'.zip'))
    Write-Host "Private evidence ZIP: $session.zip"
}
exit $code
