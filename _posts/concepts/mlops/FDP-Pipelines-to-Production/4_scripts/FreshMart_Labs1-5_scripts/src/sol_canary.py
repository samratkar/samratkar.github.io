# CHALLENGE solution: a canary rollout decided on the BUSINESS signal, with enough
# evidence to decide.
#
# The naive canary simulator compares one round of production against one round of
# canary and announces a verdict. At 5% of a thousand requests that is fifty
# canary exposures, and fifty exposures cannot distinguish a 21% conversion rate
# from a 19% one - the same candidate will look catastrophic in one round and
# excellent in the next. A canary that decides on noise is worse than no canary,
# because it launches bad models and kills good ones with equal confidence.
#
# So: accumulate exposures across rounds, and do not decide until a two-proportion
# test has the power to say something. The business signal here is whether the
# planner acted on the recommendation and the line actually needed replenishing.
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
PROD_RATE, CAND_RATE = 0.210, 0.185      # the candidate really is worse
TRAFFIC_PER_ROUND, CANARY_PCT = 1000, 0.05
MIN_CANARY_N, ALPHA, HARM_LIMIT = 300, 0.05, -0.02


def two_proportion_p(x1, n1, x2, n2):
    """One-sided test: is the canary rate below the production rate?"""
    if min(n1, n2) == 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else float(norm.cdf((p2 - p1) / se))


prod_x = prod_n = cand_x = cand_n = 0
print(f"canary at {CANARY_PCT:.0%} of {TRAFFIC_PER_ROUND:,} requests per round")
print(f"{'round':>6}{'canary n':>10}{'prod rate':>11}{'canary rate':>13}{'delta':>9}"
      f"{'p':>8}   decision")
for rnd in range(1, 9):
    n_can = int(TRAFFIC_PER_ROUND * CANARY_PCT)
    n_prod = TRAFFIC_PER_ROUND - n_can
    prod_x += rng.binomial(n_prod, PROD_RATE); prod_n += n_prod
    cand_x += rng.binomial(n_can, CAND_RATE);  cand_n += n_can

    p_prod, p_cand = prod_x / prod_n, cand_x / cand_n
    delta = p_cand - p_prod
    pval = two_proportion_p(prod_x, prod_n, cand_x, cand_n)

    if cand_n < MIN_CANARY_N:
        decision = f"HOLD - only {cand_n} exposures, cannot decide yet"
    elif pval < ALPHA and delta < HARM_LIMIT:
        decision = "ROLLBACK - canary is worse, significantly"
    elif delta >= 0:
        decision = "PROMOTE - canary is no worse"
    else:
        decision = "HOLD - trending worse, not yet conclusive"
    print(f"{rnd:>6}{cand_n:>10}{p_prod:>11.3f}{p_cand:>13.3f}{delta:>9.3f}"
          f"{pval:>8.3f}   {decision}")
    if decision.startswith(("ROLLBACK", "PROMOTE")):
        break

print(f"\nThe truth: production converts at {PROD_RATE:.1%}, the candidate at {CAND_RATE:.1%}.")
print("A single round of 50 exposures could not have found that. Accumulated exposure")
print("could - and it cost 5% of traffic, not 100%. The AUC never entered the decision.")
