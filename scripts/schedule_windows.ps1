<#
.SYNOPSIS
Generates the scheduled task objects for the daily NWP blend pipeline.
Does NOT register the task automatically.

.DESCRIPTION
This script sets up the action, trigger, and settings for the Windows Task Scheduler.
It configures it to run daily at a configurable time (default 6:00 AM).
It includes `StartWhenAvailable` so that if your laptop is asleep or offline at 6:00 AM, 
Windows will automatically run the task as soon as the laptop wakes up and connects.

If the laptop is entirely offline during the wake period, the real-data fetches will 
naturally time out, and the pipeline will either fall back to partial data or fail safely.

To register this task yourself, run this script and pipe the output to Register-ScheduledTask, e.g.:
  .\schedule_windows.ps1
  Register-ScheduledTask -InputObject $task -TaskName "MeghaDrishtiDailyPipeline"
#>

param (
    [string]$RunTime = "06:00am",
    [string]$PythonPath = "python",
    [string]$WorkingDir = "C:\dev\megha-drishti"
)

$action = New-ScheduledTaskAction -Execute $PythonPath -Argument "scripts\run_daily.py --date latest" -WorkingDirectory $WorkingDir
$trigger = New-ScheduledTaskTrigger -Daily -At $RunTime
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Write-Host "Task configuration created. It will run daily at $RunTime."
Write-Host "If the laptop is asleep, it will run as soon as it wakes up (StartWhenAvailable)."
Write-Host "To register the task, run:"
Write-Host "Register-ScheduledTask -Action `$action -Trigger `$trigger -Settings `$settings -TaskName `"MeghaDrishtiDailyPipeline`""

# Return the objects so the user can inspect or register them
$taskParams = @{
    Action = $action
    Trigger = $trigger
    Settings = $settings
}
return $taskParams
