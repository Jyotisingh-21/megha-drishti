$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\dev\megha-drishti\scripts\run_daily.py --date latest" -WorkingDirectory "C:\dev\megha-drishti"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "MeghaDrishtiDailyPipeline" -Description "Runs the daily NWP blend pipeline"
