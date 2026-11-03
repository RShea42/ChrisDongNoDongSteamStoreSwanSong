"""Host the Streamlit EDA dashboard on Modal.

One-time: modal secret create steam-eda-supabase SUPABASE_DB_URL=...
Deploy:   modal deploy modal_app.py
"""
import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .add_local_file(HERE / "app.py", "/root/app.py")
)

app = modal.App("steam-eda-dashboard", image=image)


@app.function(
    secrets=[modal.Secret.from_name("steam-eda-supabase")],
    scaledown_window=15 * 60,  # stay warm 15 min after the last visitor so the data cache survives
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=100)  # one container serves every visitor, sharing Streamlit's cache
@modal.web_server(PORT, startup_timeout=120)
def dashboard():
    cmd = (f"streamlit run /root/app.py --server.port {PORT} --server.address 0.0.0.0 --server.headless true "
           "--server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false")
    subprocess.Popen(shlex.split(cmd))
