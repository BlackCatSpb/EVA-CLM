@echo off
cd /d "C:\Users\black\OneDrive\Desktop\WideBind"
echo [WideBind] Loading environment...
call conda activate base 2>nul || echo [WideBind] No conda, using system Python

echo [WideBind] Starting training...
echo   Data: C:\Users\black\OneDrive\Desktop\fcp
echo   Model: 152M params, 32 layers, B=2, L=128
echo   VRAM: ~2-3 GB peak
echo.

echo   TRAINING ON CLEAN DATA (PAD REMOVED) - STARTING FRESH
echo.

start /b /wait "" python train.py ^
    --data-dir "C:\Users\black\OneDrive\Desktop\fcp" ^
    --save-dir checkpoints ^
    --batch-size 2 ^
    --seq-len 128 ^
    --n-layers 24 ^
    --bottleneck 896 ^
    --bind-K 64 ^
    --mlp-groups 8 ^
    --mlp-expand 8 ^
    --lr 3e-4 ^
    --max-steps 500000 ^
    --warmup 1000 ^
    --log-interval 100 ^
    --eval-interval 1000 ^
    --save-interval 5000 ^
    --scheduler mirror

echo.
echo [WideBind] Training finished.
pause
