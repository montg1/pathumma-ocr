# Start the Pathumma Vision OCR server (http://127.0.0.1:8000, Swagger UI at /docs)
# Optional env overrides: $env:PORT, $env:HOST, $env:MAX_PIXELS, $env:DEFAULT_MAX_TOKENS
$env:PYTHONIOENCODING = "utf-8"
& "$env:USERPROFILE\.venvs\pathumma-vision\Scripts\python.exe" "$PSScriptRoot\server.py"
