"""FDP setup check - run from ~/freshmart with .venv active."""
import sys, shutil, subprocess, importlib.metadata as md

ok = True
def report(label, good, detail=""):
    global ok
    ok &= good
    print(f"[{'OK ' if good else 'FIX'}] {label} {detail}")

report("Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0])
for pkg in ["pandas", "numpy", "pyarrow", "seaborn", "plotly", "altair", "ydata-profiling",
            "prefect", "dvc", "great-expectations", "feast", "scikit-learn", "xgboost",
            "mlflow", "fastapi", "uvicorn", "requests", "streamlit", "scipy"]:
    try:
        report(pkg, True, md.version(pkg))
    except md.PackageNotFoundError:
        report(pkg, False, "not installed")
for tool in ["git", "dvc", "feast", "docker"]:
    report(f"{tool} on PATH", shutil.which(tool) is not None)
if shutil.which("docker"):
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    report("Docker daemon running", r.returncode == 0)
print("\nALL GOOD - you are ready" if ok else "\nFix the [FIX] lines, then run again")
