# CHALLENGE solution: prove that rollback is a pointer move, not a rebuild.
#
# This runs against the REAL registry, not a toy class. It moves @production
# between actual model versions and asks the running service which one it is
# serving, so the claim "rollback is instant" is demonstrated rather than
# asserted.
#
# The one honest caveat, and it is the interesting part: the alias move is
# instant, but this service reads the alias once, at startup. So the pointer moves
# in milliseconds and the fleet turns over as fast as it can restart. That is
# still far faster than a rebuild - and it is why the deployment strategy has to
# say how the restart happens, not just that the alias moved.
import subprocess, sys, time
import mlflow, requests

NAME, PORT = "freshmart_reorder_model", 8077
mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
c = mlflow.MlflowClient()


def alias(a):
    try:
        return c.get_model_version_by_alias(NAME, a).version
    except Exception:
        return None


def serve_and_ask():
    """Start the API against @production and ask it what it is serving."""
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.api:app",
                             "--port", str(PORT)],
                            env={**__import__("os").environ, "MODEL_ALIAS": "production"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                r = requests.get(f"http://localhost:{PORT}/ready", timeout=2)
                if r.status_code == 200:
                    return r.json()["model_version"]
            except requests.RequestException:
                pass
            time.sleep(1)
        return None
    finally:
        proc.terminate(); proc.wait(timeout=15)


blue = alias("production")
green = alias("candidate")          # Lab 3 registered the XGBoost as @candidate
print(f"blue (live)      @production = v{blue}")
print(f"green (candidate) @candidate = v{green}")
print(f"service is serving v{serve_and_ask()}\n")

# --- deploy green: record the rollback pointer FIRST, then switch -------------
c.set_registered_model_alias(NAME, "previous", blue)
c.set_registered_model_alias(NAME, "production", green)
print(f"deploy green:    @previous -> v{blue}, @production -> v{green}")
print(f"service is serving v{serve_and_ask()} after restart\n")

# --- green misbehaves: rollback is ONE alias assignment -----------------------
t0 = time.perf_counter()
c.set_registered_model_alias(NAME, "production", alias("previous"))
ms = (time.perf_counter() - t0) * 1000
print(f"rollback:        @production -> v{alias('production')}   "
      f"(one alias move, {ms:.0f} ms)")
print(f"service is serving v{serve_and_ask()} after restart")

assert alias("production") == blue
print("\nNo image was rebuilt, no model file was copied, no code was redeployed.")
print("The old version was never deleted, which is what makes revert instant -")
print("and is why 'delete the old model to save space' is a production incident.")
