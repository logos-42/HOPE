# 测试检查点功能

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "测试 HOPE 模型检查点功能" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$env:RUST_LOG = "info"

# 测试1：训练并保存检查点
Write-Host "测试1：训练10步并保存检查点..." -ForegroundColor Yellow
.\target\release\hope-train.exe train --config examples\config_with_checkpoint.json

Write-Host ""
Write-Host "检查点文件：" -ForegroundColor Green
Get-ChildItem checkpoints\*.json | Select-Object Name, Length, LastWriteTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "测试完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

