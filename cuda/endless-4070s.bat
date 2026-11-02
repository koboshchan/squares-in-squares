@echo off
title Squares endless 4070S
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/lucas/squares-cuda-poc/full-ef21e71 && .venv/bin/python -u cuda/endless.py --directory outputs/full-4070s --incumbent inputs/incumbent-safe.csv"
echo.
echo Loop exited. Read outputs\full-4070s\worker.log for errors. Double-click to resume.
pause
