# Register-ScheduledTask for NWP Blending System
# We want to match 00Z and 12Z runs.
# ECMWF 00Z is available ~07:30 UTC -> 13:00 IST (UTC+5:30).
# ECMWF 12Z is available ~19:30 UTC -> 01:00 IST (next day).
# We add a small buffer (30 mins) to be safe: 13:30 IST and 01:30 IST.

$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/run_daily.py --date latest" -WorkingDirectory "C:\dev\megha-drishti"

# 13:30 IST trigger
$trigger1 = New-ScheduledTaskTrigger -Daily -At 1:30PM
# 01:30 IST trigger
$trigger2 = New-ScheduledTaskTrigger -Daily -At 1:30AM

# Settings
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -Action $action -Trigger @($trigger1, $trigger2) -Settings $settings -TaskName "MeghaDrishti_DailyRun" -Description "NWP Pipeline Live Fetch"
Write-Output "Task MeghaDrishti_DailyRun successfully registered. Runs daily at 13:30 and 01:30 IST."
