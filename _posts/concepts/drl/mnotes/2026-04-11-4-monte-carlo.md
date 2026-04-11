---
tags : [drl, rl]
title: "Chapter 4: Monte Carlo"
category: "drl"
subcategory: "monte-carlo"
---

### The Big Picture
- **Monte Carlo (MC)**: model-free, sample-based learning from **complete episodes**.
- Unlike **Dynamic Programming**, MC does **not** need the transition model $P(s', r \mid s, a)$.
- Unlike **Temporal-Difference (TD)**, MC does **not** bootstrap from an estimated next value.
  - DP: expectation-based, model known.
  - MC: sample-based, no model, no bootstrapping.
  - TD: sample-based, no model, **with** bootstrapping.

<div class="mermaid">
flowchart TB
  A["Bellman Equations and Returns<br/>RL foundation"]
  A --> B{"Do we know the model?"}

  B -->|Yes| C["Dynamic Programming<br/>Expectation updates"]
  B -->|No| D["Model-free learning<br/>From sampled experience"]

  D --> E{"Do we update only<br/>after full episodes?"}
  E -->|Yes| F["Monte Carlo<br/>Use complete return G_t"]
  E -->|No| G["Temporal-Difference<br/>Bootstrap using V(S_t+1) or Q(S_t+1,A_t+1)"]

  F --> H["Prediction<br/>Estimate V_pi or Q_pi"]
  F --> I["Control<br/>Improve policy from sampled returns"]

  I --> J["On-policy MC<br/>epsilon-soft / epsilon-greedy"]
  I --> K["Off-policy MC<br/>importance sampling"]

  classDef mc fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;
  classDef td fill:#fff3e0,stroke:#ef6c00,stroke-width:1px;
  classDef dp fill:#e3f2fd,stroke:#1565c0,stroke-width:1px;

  C:::dp
  F:::mc
  G:::td
</div>

### Source Code Monte Carlo

<a href="https://github.com/samratkar/samratkar.github.io/blob/main/assets/drl/webinars/montecarlo/src/monte_carlo_case_study.py" target="_blank" rel="noopener noreferrer">
    Open source code for Monte Carlo case study
</a> 

This script implements **first-visit on-policy Monte Carlo control** with an
**epsilon-soft** policy on the same 3x3 stochastic gridworld used across the
other DRL case studies.

### The gridboard game 

<a href="/assets/drl/webinars/montecarlo/src/monte_carlo_game.html" target="_blank" rel="noopener noreferrer">
    Open the game in a new tab full screen mode
</a> 

The game view lets you:

- inspect the learned $Q(s,a)$ values,
- see the final epsilon-soft action probabilities,
- replay policy-driven episodes,
- and view the moving-average training curve.

### Model and Tables

<a href="/assets/drl/webinars/montecarlo/src/monte_carlo_model.html" target="_blank" rel="noopener noreferrer">
    Open the Monte Carlo model and tables view
</a> 

This view is useful when you want the tabular details directly:

- learned action-value table,
- final policy probabilities,
- visit counts for each state-action pair,
- and the transition model $P(s', r \mid s, a)$ of the gridworld.

### Notebook

<a href="https://github.com/samratkar/samratkar.github.io/blob/main/assets/drl/webinars/montecarlo/src/monte_carlo_case_study.ipynb" target="_blank" rel="noopener noreferrer">
    Open the Monte Carlo notebook
</a> 

The notebook is a compact runnable companion for the chapter. It calls the same
Monte Carlo implementation and can regenerate the exported JSON artifact used by
the browser views.


### What Monte Carlo Means

Monte Carlo methods learn by **running episodes**, observing rewards, and then averaging the
actual returns obtained. Instead of asking:

- "What is the expected next value under the full model?" as in DP,

MC asks:

- "What return did I actually observe after visiting this state or taking this action?"

If an episode generates rewards

$$
R_{t+1}, R_{t+2}, \dots, R_T
$$

then the return from time $t$ is

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T
$$

MC uses sampled values of $G_t$ to estimate:

- **State value**: $V_\pi(s)$
- **Action value**: $Q_\pi(s,a)$

This makes MC natural when:

- episodes terminate,
- simulation is available,
- the model is unknown,
- and waiting until the end of the episode is acceptable.


### Core Idea: Learn From Returns

For a fixed policy $\pi$, Monte Carlo prediction estimates value functions by averaging returns.

For state values:

$$
V_\pi(s) \approx \text{average of returns observed after visits to } s
$$

For action values:

$$
Q_\pi(s,a) \approx \text{average of returns observed after visits to } (s,a)
$$

Over time, if enough samples are collected, these averages converge to the true values under mild conditions.


### First-Visit vs Every-Visit Monte Carlo

There are two standard ways to average returns.

#### 1. First-Visit MC
- For each episode, use only the **first time** a state (or state-action pair) appears.
- Compute the return from that first occurrence onward.
- Average these returns across episodes.

#### 2. Every-Visit MC
- Use **every occurrence** of the state (or state-action pair) in the episode.
- Compute the return from each occurrence.
- Average all of them.

Both are valid. First-visit MC is usually introduced first because it is simpler to reason about.


### Monte Carlo Prediction

Suppose policy $\pi$ is fixed. Then the procedure is:

1. Generate an episode using $\pi$.
2. For each visited state $s$, compute the return $G_t$ from that point onward.
3. Update the estimate of $V_\pi(s)$ by averaging returns.

Incremental form:

$$
V(s) \leftarrow V(s) + \alpha \bigl(G_t - V(s)\bigr)
$$

This looks like a stochastic approximation update:

- target = sampled return $G_t$
- estimate = current value $V(s)$
- error = $G_t - V(s)$

Similarly for action values:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl(G_t - Q(s,a)\bigr)
$$


### Why MC Needs Episodes

MC needs a **complete return** $G_t$, so it must wait until the episode ends.

That creates two important consequences:

1. MC is naturally suited for **episodic tasks**.
2. MC updates can have **high variance**, because full returns can vary a lot from one episode to another.

This is the main tradeoff:

- MC is unbiased with respect to sampled returns.
- But MC can be noisy because it uses the full random outcome of the episode.


### Monte Carlo Control

Prediction evaluates a policy. Control improves it.

The MC control loop is:

1. Start with a policy $\pi$.
2. Generate episodes following $\pi$.
3. Estimate $Q_\pi(s,a)$ from returns.
4. Improve the policy to be greedy or nearly greedy with respect to $Q$.
5. Repeat.

This mirrors policy iteration:

- **Policy evaluation**: estimate $Q_\pi$
- **Policy improvement**: update $\pi$ using current $Q$

Because the model is unknown, both steps are done from sampled episodes rather than exact Bellman expectations.


### Exploring Starts

One classic MC control assumption is **exploring starts**:

- every state-action pair has a non-zero probability of being selected as the starting pair.

This guarantees enough exploration in theory, but it is often unrealistic in practice because
we usually cannot reset the environment to any arbitrary state-action pair.

So in practical RL, we usually prefer soft exploration policies instead.


### On-Policy Monte Carlo Control

In on-policy MC control:

- the same policy is used for both **behavior** and **improvement**.
- we commonly use an **epsilon-greedy** or **epsilon-soft** policy.

Why?

- Pure greedy action selection may stop exploring too early.
- Epsilon-greedy keeps trying non-greedy actions with small probability.

Typical rule:

- with probability $1-\epsilon$, choose the greedy action
- with probability $\epsilon$, choose a random action

This ensures continued exploration while gradually favoring better actions.


### Off-Policy Monte Carlo

Off-policy learning separates:

- **behavior policy** $b$: the policy that generates episodes
- **target policy** $\pi$: the policy we want to evaluate or improve

Because episodes come from $b$ but values are for $\pi$, we must correct the mismatch using
**importance sampling**.

Importance sampling ratio:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}
$$

Then weighted returns are used to estimate values under $\pi$.

Why this matters:

- it allows learning about one policy while following another,
- which is a key idea later in advanced RL,
- but the variance can become very large.


### MC vs DP vs TD

| Method | Model needed | Learns from samples | Bootstrapping | Update timing |
|---|---|---|---|---|
| DP | Yes | No | Yes | During sweeps over model |
| MC | No | Yes | No | After full episode |
| TD | No | Yes | Yes | Every step |

Interpretation:

- **DP** is exact planning when the environment is known.
- **MC** is pure learning from complete experience.
- **TD** is a hybrid that learns from samples but bootstraps from current estimates.


### Strengths of Monte Carlo

1. Does not require the environment model.
2. Conceptually simple: estimate values by averaging returns.
3. Works directly from real or simulated experience.
4. Useful for episodic tasks.
5. Does not bootstrap, so targets are actual sampled returns.


### Limitations of Monte Carlo

1. Must wait until the episode terminates before updating.
2. Not naturally suited to continuing tasks without modification.
3. Returns can have high variance, making learning unstable or slow.
4. Pure exploring starts is often impractical.
5. Off-policy MC with importance sampling can become very high variance.


### Intuition To Remember

- DP asks: "What should happen according to the model?"
- MC asks: "What actually happened over the whole episode?"
- TD asks: "What happened now, and what do I currently estimate will happen next?"

That single contrast is enough to place MC correctly inside the larger RL picture.


### Key Points - 
1. Monte Carlo is a **model-free** method that learns from **complete episodes**.
2. MC estimates values using the sampled return $G_t$, not the Bellman expectation over a known model.
3. MC does **not** bootstrap; its targets are actual returns observed after an episode unfolds.
4. First-visit MC uses only the first occurrence of a state or state-action pair in an episode, while every-visit MC uses all occurrences.
5. MC prediction evaluates a fixed policy by averaging returns for visited states or state-action pairs.
6. MC control alternates between estimating $Q_\pi(s,a)$ and improving the policy to be greedy or epsilon-greedy with respect to that estimate.
7. On-policy MC commonly uses epsilon-soft exploration so that all actions continue to be sampled.
8. Off-policy MC learns about a target policy from data generated by a different behavior policy using importance sampling.
9. MC is:
- model-free
- sample-based
- episode-based
- no bootstrapping
- higher variance than TD
- especially natural for episodic problems
