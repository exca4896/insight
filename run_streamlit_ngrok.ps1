Param(
  [string]$AppPath = "combine_app.py",
  [int]$Port = 8501,
  [string]$NgrokCmd = "ngrok"
)

try {
  Write-Host "Starting Streamlit..." -ForegroundColor Cyan
  $streamlit = Start-Process -FilePath "streamlit" -ArgumentList "run", $AppPath, "--server.port", $Port, "--server.address", "0.0.0.0" -PassThru

  Start-Sleep -Seconds 1

  Write-Host "Starting ngrok tunnel..." -ForegroundColor Cyan
  $ngrok = Start-Process -FilePath $NgrokCmd -ArgumentList "http", $Port -NoNewWindow -PassThru

  # Wait for ngrok local API and fetch public URL
  $publicUrl = $null
  $tries = 0
  while (-not $publicUrl -and $tries -lt 30) {
    try {
      $resp = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
      if ($resp.tunnels -and $resp.tunnels.Count -gt 0) {
        $publicUrl = $resp.tunnels[0].public_url
        break
      }
    } catch {}
    Start-Sleep -Seconds 1
    $tries++
  }

  if ($publicUrl) {
    Write-Host "ngrok public URL: $publicUrl" -ForegroundColor Green
    Start-Process $publicUrl
  } else {
    Write-Warning "Could not fetch ngrok public URL (timeout). Check ngrok output."
  }

  Write-Host "Tunnel running. Press Ctrl+C in this window to stop both processes." -ForegroundColor Yellow
  # Wait for ngrok process to exit; when it does, kill Streamlit too
  Wait-Process -Id $ngrok.Id
}
finally {
  Write-Host "`nCleaning up processes..." -ForegroundColor Cyan
  if ($streamlit -and (Get-Process -Id $streamlit.Id -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $streamlit.Id -Force -ErrorAction SilentlyContinue
  }
  if ($ngrok -and (Get-Process -Id $ngrok.Id -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $ngrok.Id -Force -ErrorAction SilentlyContinue
  }
  Write-Host "Stopped." -ForegroundColor Green
}