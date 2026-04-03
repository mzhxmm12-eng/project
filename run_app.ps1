$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot
$env:DEEPSEEK_API_KEY = "sk-369ec6d1b22f4fa1aabdbff7b65d7861"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:NO_PROXY = "localhost,127.0.0.1,api.deepseek.com"
$env:no_proxy = "localhost,127.0.0.1,api.deepseek.com"

$fastApiProcess = Start-Process `
  -FilePath "python" `
  -ArgumentList "-m", "uvicorn", "app.chat_api:app", "--port", "8502", "--host", "0.0.0.0" `
  -PassThru `
  -WindowStyle Hidden

try {
  python -m streamlit run app/streamlit_app.py --server.port 8501
}
finally {
  if ($fastApiProcess -and -not $fastApiProcess.HasExited) {
    Stop-Process -Id $fastApiProcess.Id -Force
  }
}
