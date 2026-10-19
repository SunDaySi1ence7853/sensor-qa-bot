# check.ps1 —— push 前三道闸门：pytest 全绿 → demo 不崩 → 场景全对
# 规则：全文件只允许一个成功出口，且必须在最末尾。改本文件时整块替换，不要追加。


[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

function Invoke-Gate {
    param([string]$Name, [string[]]$Cmd)
    Write-Host ""
    Write-Host "===== 闸门 $Name =====" -ForegroundColor Cyan
    & $Cmd[0] $Cmd[1..($Cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 闸门 $Name 未通过（退出码 $LASTEXITCODE），禁止 push" -ForegroundColor Red
        exit 1
    }
}

Invoke-Gate "1/3 pytest 全量"   @("python", "-m", "pytest", "-q")
Invoke-Gate "2/3 动态路由 demo" @("python", "-m", "src.agents.demo_dynamic")
Invoke-Gate "3/3 场景验收"      @("python", "-m", "src.agents.scenarios")

Write-Host ""
Write-Host "✅ 三道闸门全部通过，可以 push" -ForegroundColor Green
exit 0