<#
.SYNOPSIS
Registers (or removes) the Windows scheduled task that runs daily_refresh.ps1: every time you sign in to
Windows, and once a day at a set time.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1             # at sign-in, and daily at 8:30 am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -At 7:00am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -NoSignIn   # only the daily time
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -Remove

.NOTES
The task runs as you, while you're signed in, so git can push with your saved credentials.
If the PC is off or asleep at the set time, it runs as soon as you're back. Running more than once a day is
harmless: responses are cached, and it only commits and pushes when something is new.
#>
param([string]$At = "8:30am", [switch]$NoSignIn, [switch]$Remove)

$name = "Clippers Courtside daily refresh"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    Write-Output "Removed '$name'."
    return
}

$script = Join-Path $PSScriptRoot "daily_refresh.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$triggers = @(New-ScheduledTaskTrigger -Daily -At $At)
if (-not $NoSignIn) {
    # every sign-in, two minutes later so the network is up first
    $signin = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $signin.Delay = "PT2M"
    $triggers += $signin
}
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings -Force `
    -Description "Pulls last night's Clippers games, rebuilds the analysis and pushes it so the hosted dashboard updates." |
    Out-Null
$when = if ($NoSignIn) { "daily at $At" } else { "2 minutes after each sign-in, and daily at $At" }
Write-Output "Registered '$name': $when (or as soon as the PC is on after a missed time)."
Write-Output "Logs: $(Join-Path (Split-Path -Parent $PSScriptRoot) 'logs')\refresh-<date>.log"
