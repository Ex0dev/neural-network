@echo off
call vc.bat
pushd C:\Development\C\billybobsneuralnet
cl neuralnet.c nn_common.c /EHsc /Zi /link /out:neuralnet1.exe
popd
