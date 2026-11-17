param([Parameter(Mandatory=$true)][string]$Output,
      [string]$TaskName = 'B17-Single-Tank-672912-Resume')
$ErrorActionPreference = 'Stop'
$Output = (Resolve-Path -LiteralPath $Output).Path
if (-not (Test-Path -LiteralPath (Join-Path $Output 'resume_command.json'))) { throw 'Missing reviewed resume command' }
$launcher = Join-Path $PSScriptRoot 'Resume-Batch.ps1'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $launcher + '" -Output "' + $Output + '"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$login = New-ScheduledTaskTrigger -AtLogOn -User $user
$periodic = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
# Do not overwrite an unrelated existing task.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { throw "Task already exists: $TaskName" }
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($login,$periodic) -Settings $settings -Principal $principal -Description 'Resume the exact B17 single native batch after interruption; stop on numerical failure or STOP file.'
