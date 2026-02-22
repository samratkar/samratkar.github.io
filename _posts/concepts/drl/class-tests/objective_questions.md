# DRL Objective Questions (Syllabus-Aligned)

## Section A: Multi-Armed Bandit (MAB)

### Q1
In a k-armed bandit setting, what does `q*(a)` represent?

A. The reward obtained at time `t` after selecting action `a`
B. The probability of choosing action `a`
C. The true expected reward of action `a`
D. The discounted return from state `s`

**Answer:** C

### Q2
Which statement best describes UCB action selection?

A. It explores by choosing random actions with fixed probability `epsilon`
B. It adds an exploration bonus that is larger for less-visited actions
C. It always chooses the action with highest immediate sampled reward
D. It requires a transition model with states and actions

**Answer:** B

### Q3
For epsilon-greedy with `epsilon = 0.2`, the agent explores with probability:

A. 0.8
B. 0.5
C. 0.2
D. Depends on number of actions only

**Answer:** C

### Q4
In non-stationary bandits, which update rule is usually preferred?

A. Sample-average update with step size `1/N(a)`
B. Constant step-size update with fixed `alpha`
C. Monte Carlo return averaging across full episodes
D. Dynamic programming policy evaluation

**Answer:** B

### Q5
Which quantity is directly observed by the agent immediately after taking action `A_t`?

A. `q*(A_t)`
B. `Q_t(A_t)`
C. `R_t`
D. `pi(A_t|S_t)`

**Answer:** C

## Section B: Markov Decision Process (MDP)

### Q6
An MDP is commonly represented as:

A. `(S, A, R)`
B. `(S, A, P, R, gamma)`
C. `(A, P, V, Q)`
D. `(S, A, T)`

**Answer:** B

### Q7
Which component specifies environment dynamics in an MDP?

A. `pi(a|s)`
B. `R(s,a,s')`
C. `P(s'|s,a)`
D. `v_pi(s)`

**Answer:** C

### Q8
If an environment is deterministic, then for any fixed `(s,a)`:

A. Multiple next states always have equal probability
B. Transition probabilities must all be less than 1
C. Exactly one next state has probability 1
D. Rewards must be stochastic

**Answer:** C

### Q9
When `gamma = 0`, the agent:

A. Considers only immediate rewards
B. Maximizes infinite-horizon returns without discount
C. Ignores immediate rewards
D. Requires episodic tasks only

**Answer:** A

### Q10
Which statement is true about policy `pi(a|s)` in an MDP?

A. It must always be deterministic
B. For each state, action probabilities sum to 1
C. It is identical to transition probability `P(s'|s,a)`
D. It is defined only for terminal states

**Answer:** B

## Section C: Value Functions and Bellman Equations

### Q11
The state-value function `v_pi(s)` is:

A. Expected return from state `s` following policy `pi`
B. Expected immediate reward only
C. Maximum action value across actions in state `s`
D. Transition probability from `s` to `s'`

**Answer:** A

### Q12
The action-value function `q_pi(s,a)` gives:

A. Expected return if action is chosen uniformly at random
B. Expected return from taking action `a` in `s`, then following `pi`
C. Probability of choosing action `a` in state `s`
D. Immediate reward independent of future states

**Answer:** B

### Q13
Which relation is correct?

A. `q_pi(s,a) = sum_a pi(a|s) v_pi(s)`
B. `v_pi(s) = max_a q_pi(s,a)` for any policy `pi`
C. `v_pi(s) = sum_a pi(a|s) q_pi(s,a)`
D. `v_pi(s) = R(s)` only

**Answer:** C

### Q14
Bellman optimality equation for state values uses which operator over actions?

A. Mean
B. Sum
C. Product
D. Max

**Answer:** D

### Q15
Increasing `gamma` (e.g., from 0.2 to 0.9) generally makes the agent:

A. More myopic
B. More far-sighted about future rewards
C. Unable to compute value functions
D. Ignore transition probabilities

**Answer:** B

## Answer Key (Quick View)

1. C  
2. B  
3. C  
4. B  
5. C  
6. B  
7. C  
8. C  
9. A  
10. B  
11. A  
12. B  
13. C  
14. D  
15. B
