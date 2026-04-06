@echo off
for /r %%i in (Main*.tex) do texify  --engine=xetex -cp %%i
