"""Start the ThermoEqui-Agent API server with DeepSeek provider."""
import os, subprocess, sys, time

os.environ["LLM_PROVIDER"] = "deepseek"
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["DEEPSEEK_MODEL"] = "deepseek-v4-flash"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "apps.api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    cwd=r"D:\Codex\ThermoEqui-Agent-main",
)
time.sleep(5)
print(f"SERVER_PID:{proc.pid}", flush=True)
proc.wait()
