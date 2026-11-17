param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
$Output = (Resolve-Path -LiteralPath $Output).Path
if (Test-Path -LiteralPath (Join-Path $Output 'STOP')) { exit 0 }
$queuePath = Join-Path $Output 'queue.json'
if (Test-Path -LiteralPath $queuePath) {
    $queue = Get-Content -Raw -Encoding UTF8 -LiteralPath $queuePath | ConvertFrom-Json
    if ($queue.status -in @('completed','stopped_with_failures','blocked')) { exit 0 }
}
$config = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $Output 'resume_command.json') | ConvertFrom-Json
Set-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$batchArguments = @($config.args)
& $config.python @batchArguments >> (Join-Path $Output 'launcher.log') 2>&1
exit $LASTEXITCODE
