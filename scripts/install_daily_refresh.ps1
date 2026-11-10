<#
.SYNOPSIS
Registers (or removes) the Windows scheduled task that runs daily_refresh.ps1 every morning.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1             # daily at 7:00 am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -At 8:30am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -Remove

.NOTES
The task runs as you, while you're signed in, so git can push with your saved credentials.
If the PC is off or asleep at the set time, it runs as soon as you're back.
#>
param([string]$At = "7:00am", [switch]$Remove)

$name = "Clippers Courtside daily refresh"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    Write-Output "Removed '$name'."
    return
}

$script = Join-Path $PSScriptRoot "daily_refresh.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force `
    -Description "Pulls last night's Clippers games, rebuilds the analysis and pushes it so the hosted dashboard updates." |
    Out-Null
Write-Output "Registered '$name': daily at $At, or as soon as the PC is on after that."
Write-Output "Logs: $(Join-Path (Split-Path -Parent $PSScriptRoot) 'logs')\refresh-<date>.log"
