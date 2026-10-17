pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "❌ pytest 未全绿，禁止 push" -ForegroundColor Red; exit 1 }
python -m src.agents.demo_dynamic
if ($LASTEXITCODE -ne 0) { Write-Host "❌ demo 崩溃，禁止 push" -ForegroundColor Red; exit 1 }
Write-Host "✅ 可以 push" -ForegroundColor Green
pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "❌ pytest 未全绿，禁止 push" -ForegroundColor Red; exit 1 }
python -m src.agents.demo_dynamic
if ($LASTEXITCODE -ne 0) { Write-Host "❌ demo 崩溃，禁止 push" -ForegroundColor Red; exit 1 }
python -m src.agents.scenarios
if ($LASTEXITCODE -ne 0) { Write-Host "❌ 场景未全对，禁止 push" -ForegroundColor Red; exit 1 }
Write-Host "✅ 可以 push" -ForegroundColor Green