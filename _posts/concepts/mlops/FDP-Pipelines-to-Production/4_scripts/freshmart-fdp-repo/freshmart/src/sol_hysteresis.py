# CHALLENGE solution: hysteresis, so one noisy window cannot trigger a re-run.
#
# A monitor that fires on every window above threshold will fire, retrain, fire
# again on the next window, and retrain again - a loop that is worse than no
# automation, because now the thrashing has a schedule. Two guards fix it:
# persistence (act only after N consecutive drifted windows) and a daily cap.
#
# The sequence below is the real one: the PSI values this monitor produced across
# hourly windows around the festival promotion, with one isolated spike from a
# short window that happened to catch a delivery burst.
from collections import deque


class DriftController:
    def __init__(self, psi_threshold=0.20, persistence=2, max_per_day=1):
        self.th, self.persist, self.max = psi_threshold, persistence, max_per_day
        self.window = deque(maxlen=persistence)
        self.reruns_today = 0

    def observe(self, psi_max):
        self.window.append(psi_max > self.th)
        persistent = len(self.window) == self.persist and all(self.window)
        if persistent and self.reruns_today < self.max:
            self.reruns_today += 1
            return "RERUN", "drift persisted"
        if persistent:
            return "HOLD", f"drift persists but daily cap ({self.max}) reached"
        if self.window[-1]:
            return "HOLD", "drifted, waiting for confirmation"
        return "HOLD", "within threshold"


ctrl = DriftController(persistence=2, max_per_day=1)
sequence = [0.05, 0.31, 0.04, 0.33, 0.81, 0.77]   # spike at t1, sustained from t3

print(f"{'window':>7}{'psi_max':>10}   action   why")
for t, p in enumerate(sequence):
    action, why = ctrl.observe(p)
    print(f"{'t' + str(t):>7}{p:>10.2f}   {action:<8} {why}")
print("\nThe lone spike at t1 is absorbed: one drifted window is noise until the next")
print("one agrees. Sustained drift from t3 triggers exactly one re-run, and the cap")
print("holds the line at t5 - the pipeline is already re-running, and asking twice")
print("would not make it finish sooner.")
