# Assignment 1 - DRL Foundations

**Course Scope:** MAB, MDP, Bellman Value Functions, and Dynamic Programming  
**Submission Format:** PDF report + code archive (if implementation option chosen)

## Question 1 (MDB/MAB)

> Interpreting "MDB" as **MAB (Multi-Armed Bandit)** based on syllabus context.

Choose **one** of the following tracks:

### Track A: Research Paper Analysis (MAB)
Select one research paper on contextual or non-stationary bandits (for example: LinUCB, Thompson Sampling for bandits, or non-stationary UCB variants) and critically analyze it.

Suggested paper references (choose one; grouped by difficulty):

Easy:
1. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Bandit chapter.
2. Bubeck, S., and Cesa-Bianchi, N. (2012). *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*.

Medium:
1. Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*.
2. Agrawal, S., and Goyal, N. (2012). *Analysis of Thompson Sampling for the Multi-armed Bandit Problem*.
3. Garivier, A., and Moulines, E. (2011). *On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems*.

Hard:
1. Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*.
2. Abbasi-Yadkori, Y., Pal, D., and Szepesvari, C. (2011). *Improved Algorithms for Linear Stochastic Bandits*.
3. Besbes, O., Gur, Y., and Zeevi, A. (2014). *Stochastic Multi-Armed-Bandit Problem with Non-Stationary Rewards*.

Your analysis must include:
1. Problem setting and why it is a bandit problem (not full MDP).
2. Exploration-exploitation strategy used in the paper.
3. Key assumptions (stationarity, reward noise, context availability).
4. Experimental setup and baselines.
5. One limitation and one practical extension for real-world deployment.

### Track B: Partial Gymnasium Implementation (MAB)
Implement a custom `gymnasium.Env` for a non-stationary k-armed bandit and compare at least two methods (e.g., epsilon-greedy vs UCB, or UCB vs Thompson Sampling).

Minimum requirements:
1. `k >= 5` arms with drifting reward means.
2. Agent update logic using constant step-size `alpha`.
3. Training for at least 10,000 steps.
4. Plot cumulative reward and regret.
5. Short discussion on which method adapts better and why.

## Question 2 (Dynamic Programming using Mach-Predictor MDP)

### Context for Students (Read Carefully)

Most students know route optimization, but this problem is different: we optimize **speed profile (Mach number)** over time for fuel efficiency while keeping flight behavior operationally valid.

#### Operational Scenario
You are designing an onboard decision-support policy for cruise segments of a transport aircraft.  
At each decision step, the policy recommends a Mach number. The aircraft then evolves to a new condition (new weight, weather disturbance, altitude trend), and the next Mach decision is made.

#### Why this is an MDP (not just a bandit)
If each Mach choice affected only immediate fuel flow, this would be bandit-like.  
In reality, each action changes the future state:
1. Fuel burn reduces aircraft weight, which changes later fuel efficiency.
2. Weather/wind and temperature deviations evolve over time.
3. Flight phase (climb/cruise/descent) and target altitude constraints affect later feasible behavior.

Because decisions influence future states and future rewards, the problem is naturally an **MDP**.

#### State Variables (continuous baseline)
Use this state template:
1. `Altitude_ft`: current pressure altitude.
2. `Weight_kg`: current aircraft gross weight.
3. `TAT_C`: total air temperature.
4. `CAS_kts`: calibrated airspeed.
5. `TempDev_C`: temperature deviation from ISA profile.
6. `Wind_kts`: along-track wind component.
7. `Phase`: encoded phase (climb/cruise/descent).
8. `TargetAlt_ft`: target altitude for current segment.

#### Action
1. Continuous action baseline: `Mach in [0.70, 0.86]`.
2. For Dynamic Programming (tabular), discretize Mach into bins (example: 0.70, 0.74, 0.78, 0.82, 0.86).

#### Transition Intuition
Given state `s_t` and chosen Mach `a_t`, next state `s_{t+1}` changes due to:
1. **Fuel burn**: weight decreases approximately by fuel-flow x delta-time.
2. **Atmospheric drift**: temperature deviation and wind vary stochastically (small random walk / AR(1)-style).
3. **Phase dynamics**: altitude moves toward `TargetAlt_ft` according to phase-specific rates.
4. **Kinematics coupling**: Mach affects TAS/CAS and therefore drag and fuel.

#### Reward Design (per decision step)
A practical reward can be modeled as:

`r_t = - FuelPerNM_t - lambda_alt * AltitudeDeviationPenalty_t - lambda_energy * EnergyPenalty_t`

Where:
1. `FuelPerNM_t` encourages fuel-efficient operation.
2. `AltitudeDeviationPenalty_t` penalizes not tracking target altitude profile.
3. `EnergyPenalty_t` penalizes unrealistic/high-energy behavior (for example, excessive Mach in climb/descent or low-altitude overspeed).

Maximizing cumulative discounted return is equivalent to minimizing long-term operational cost under these penalties.

#### Recommended Modeling Thought Process
Use this process in your report:
1. Define objective in engineering terms: minimize fuel per distance while respecting operational behavior.
2. Identify decision variable: Mach command at each step.
3. Identify state memory: variables needed so next dynamics and cost are conditionally determined.
4. Add uncertainty sources: wind and temperature evolution.
5. Encode constraints via penalties (or action/state limits).
6. Convert continuous problem to tractable DP form via discretization (for tabular methods).
7. Solve Bellman updates and interpret resulting policy trends (e.g., higher Mach may be preferred only in specific high-weight/high-headwind regimes).

Use the **Aircraft Mach Optimization** problem statement as your MDP base:
1. State (continuous): altitude, weight, TAT, CAS, temperature deviation, wind, phase, target altitude
2. Action: Mach number in `[0.70, 0.86]`
3. Reward: negative fuel-per-distance with penalties for altitude deviation and poor energy management

Choose **one** of the following tracks:

### Track A: Research Paper Analysis (DP/Planning)
Pick one paper or technical report on model-based RL or dynamic programming style planning for continuous control, and map its method to the mach-predictor MDP.

#### Intent for Track A (Paper-Only)
For Track A, your work must be a **pure research paper analysis**.  
You are **not required to implement** the Mach environment, train an agent, or code a solver.

Role of the Mach problem in Track A:
1. It is a **case-study anchor** used to test whether you can transfer theory to a realistic decision-making setting.
2. You must map the paper's assumptions and method components to this specific problem.
3. You must evaluate practical feasibility, limitations, and needed adaptations for aviation-style deployment.

Track A expectations:
1. Explain the paper's core method clearly and correctly.
2. Map method components to Mach problem elements (state, action, transition, reward, constraints).
3. Critically assess what works, what fails, and what must be modified.
4. Propose a concrete adaptation path for the Mach setting.

Track A non-expectations:
1. No mandatory coding or training runs.
2. No purely generic summary disconnected from the Mach case-study.

Suggested paper references (choose one; grouped by difficulty):

Easy:
1. Moerland, T. M., Broekens, J., Plaat, A., and Jonker, C. M. (2023). *Model-based Reinforcement Learning: A Survey*.
2. Sutton, R. S. (1991). *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*.
3. Nagabandi, A., Kahn, G., Fearing, R. S., and Levine, S. (2018). *Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning*.

Medium:
1. Chua, K., Calandra, R., McAllister, R., and Levine, S. (2018). *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)*.
2. Riedmiller, M. (2005). *Neural Fitted Q Iteration -- First Experiences with a Data Efficient Neural Reinforcement Learning Method*.
3. Ernst, D., Geurts, P., and Wehenkel, L. (2005). *Tree-Based Batch Mode Reinforcement Learning*.

Hard:
1. Deisenroth, M. P., and Rasmussen, C. E. (2011). *PILCO: A Model-Based and Data-Efficient Approach to Policy Search*.
2. Tassa, Y., Erez, T., and Todorov, E. (2012). *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*.
3. Munos, R., and Szepesvari, C. (2008). *Finite-Time Bounds for Fitted Value Iteration*.

Your analysis must include:
1. How continuous states/actions are handled (discretization, function approximation, MPC, etc.).
2. Bellman backup form used (value iteration, policy iteration, fitted value iteration, etc.).
3. Computational trade-offs for aviation-scale deployment.
4. Safety/constraint handling (altitude and speed constraints).
5. A proposed adaptation plan for the mach-predictor setup.

### Track B: Partial Gymnasium + Dynamic Programming Prototype
Build a simplified tabular approximation of the mach-predictor problem and solve it with **Value Iteration** or **Policy Iteration**.

Minimum requirements:
1. Create a discretized state space (example: altitude bands x weight bands x phase).
2. Discretize Mach into at least 5 actions.
3. Define transition and reward model consistent with fuel minimization objective.
4. Implement Value Iteration or Policy Iteration from scratch.
5. Report: final policy table, convergence plot, and 2 failure cases of discretization.

#### Hints and Reading for 2B (Non-Aerospace Background)

If you are not from an aerospace background, use a control-systems view: this is a sequential speed-selection problem with penalties, not an aircraft design task.

Suggested starter simplification:
1. State bins: `AltitudeBand x WeightBand x Phase = 3 x 3 x 3 = 27 states`.
2. Action bins: `Mach = {0.70, 0.74, 0.78, 0.82, 0.86}`.
3. Target altitude band: choose one target for each phase segment.

Simple transition hints:
1. Weight tends to reduce over time; probability of reduction can increase with higher Mach.
2. Climb phase tends to move altitude band up; descent phase tends to move it down; cruise tends to stay.
3. Add small stochasticity so transitions are not fully deterministic.

Simple reward hints:
1. Use cost-shaped reward: `r = -(fuel_cost + altitude_penalty + phase_penalty)`.
2. Example fuel proxy: `fuel_cost = base_phase_cost + c1 * mach^2 + c2 * heavy_weight_indicator`.
3. Add altitude tracking penalty for distance from target altitude band.
4. Add phase-consistency penalty for very high Mach in climb/descent.

DP implementation hints:
1. Start with Value Iteration first (usually easier to debug than full Policy Iteration).
2. Use a convergence threshold such as `max_s |V_new(s) - V_old(s)| < 1e-4`.
3. Validate transition rows sum to 1 for every `(s,a)`.

Failure-case hints (for report requirement):
1. Coarse discretization failure: policy becomes insensitive and unrealistic.
2. Over-fine discretization failure: sparse transitions, unstable values, or very slow convergence.

Recommended beginner literature:
1. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (Chapters 3 and 4).
2. Puterman, M. L. (1994). *Markov Decision Processes* (selected sections on value/policy iteration).
3. OpenAI Spinning Up documentation (RL foundations and value methods overview).
4. NASA Beginner's Guide to Aeronautics (basic intuition on speed, drag, and fuel effects).

## Expected Deliverables
1. `report.pdf` (6-10 pages, with equations/plots/tables as applicable).
2. Source code with README (for Track B).
3. Reproducibility notes: dependencies, command to run, and seed used.

## Evaluation Rubric (Indicative)
1. Conceptual correctness: 30%
2. Methodological rigor: 25%
3. Experimental quality / analysis depth: 25%
4. Clarity and reproducibility: 20%
