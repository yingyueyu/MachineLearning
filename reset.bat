@echo off

setlocal
:PROMPT
SET /P AREYOUSURE=ȷ������ô (Y/[N])?
IF /I "%AREYOUSURE%" NEQ "Y" GOTO END

echo ���ÿ�ʼ
git reset --hard HEAD~1
echo ���ý���

:END
endlocal
pause
