@echo off
call vc.bat
cd C:\Development\C
cl neuralnet.c nn_common.c /EHsc /Zi /link /out:neuralnet1.exe
