<#
.SYNOPSIS
Registers (or removes) the Windows scheduled task that runs daily_refresh.ps1 as soon as the PC is back in
use (sign-in, unlock, or waking from sleep) and once a day at a set time.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1             # on sign-in/unlock/wake, and daily at 8:30 am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -At 7:00am
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -NoSignIn   # only the daily time (no sign-in/unlock/wake)
powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh.ps1 -Remove

.NOTES
The task runs as you, while you're signed in, so git can push with your saved credentials.
If the PC is off or asleep at the set time, it runs as soon as you're back. Running more than once a day is
harmless: responses are cached, it only commits and pushes when something is new, and a run within 30 minutes
of the last one is skipped (unlocking a few times in a row doesn't refresh a few times).
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
    # Run as soon as the PC is back in use; daily_refresh.ps1 waits for the network itself.
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $signin = New-ScheduledTaskTrigger -AtLogOn -User $user  # turning the PC on
    $signin.Delay = "PT30S"
    $triggers += $signin
    $ns = "Root/Microsoft/Windows/TaskScheduler"
    # unlocking it (coming back to it, or waking from sleep to the lock screen); 8 = session unlock
    $unlock = New-CimInstance -CimClass (Get-CimClass -Namespace $ns -ClassName MSFT_TaskSessionStateChangeTrigger) `
        -ClientOnly -Property @{ StateChange = [uint32]8; UserId = $user; Delay = "PT30S"; Enabled = $true }
    $triggers += $unlock
    # waking from sleep without a lock screen: Windows logs Power-Troubleshooter event 1
    $query = '<QueryList><Query Id="0" Path="System"><Select Path="System">' +
        "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]" +
        '</Select></Query></QueryList>'
    $wake = New-CimInstance -CimClass (Get-CimClass -Namespace $ns -ClassName MSFT_TaskEventTrigger) `
        -ClientOnly -Property @{ Subscription = $query; Delay = "PT30S"; Enabled = $true }
    $triggers += $wake
}
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings -Force `
    -Description "Pulls last night's Clippers games, rebuilds the analysis and pushes it so the hosted dashboard updates." |
    Out-Null
$when = if ($NoSignIn) { "daily at $At" } else { "30 seconds after each sign-in, unlock or wake from sleep, and daily at $At" }
Write-Output "Registered '$name': $when (or as soon as the PC is on after a missed time)."
Write-Output "Logs: $(Join-Path (Split-Path -Parent $PSScriptRoot) 'logs')\refresh-<date>.log"
