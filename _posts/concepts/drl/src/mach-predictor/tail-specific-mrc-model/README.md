# Tail-Specific MRC Modeling

## Purpose

This document defines a scientifically grounded approach for estimating tail-specific Maximum Range Cruise (MRC) speed from QAR data.

The recommended approach is:

1. Treat MRC as a performance-optimization problem, not as a direct label prediction problem.
2. Build a gray-box aircraft performance model with fleet-level shared structure.
3. Add tail-specific calibrated parameters to capture airframe and engine variability.
4. Optionally add a small residual data-driven correction model.
5. Compute MRC by direct bounded optimization over Mach.

This is the preferred approach over PPO/RL, PDE-first modeling, or a pure PINN for the present problem.

## Problem Statement

Given QAR-derived cruise state information for a specific aircraft tail, estimate the tail-specific MRC Mach.

In still air, MRC is the speed that maximizes air range per unit fuel:

```text
MRC(x, tail) = argmax_M SR(x, M, tail)
```

where:

```text
SR(x, M, tail) = TAS(x, M) / FuelFlow(x, M, tail)
```

If the operational objective is ground range rather than still-air range, then:

```text
MRC_ground(x, tail) = argmax_M GR(x, M, tail)
GR(x, M, tail) = GS(x, M) / FuelFlow(x, M, tail)
```

The choice between still-air MRC and wind-aware operational optimum must be made explicitly. If the objective is a canonical aerodynamic recommendation, still-air MRC is cleaner. If the objective is dispatch usefulness, ground-range optimization may be preferable.

## Why This Is Not a Direct Supervised Labeling Problem

QAR usually records the speed that was flown, not the speed that was optimal for maximum range. The flown speed is affected by:

- airline cost index
- schedule pressure
- ATC constraints
- weather
- route structure
- pilot/FMS operational choices
- maintenance condition

Therefore, the target is not:

```text
state -> flown Mach
```

The correct scientific target is:

```text
state + tail -> performance model -> optimized Mach
```

This is why the model should predict fuel/performance as a function of state and Mach, and then derive MRC through optimization.

## Recommended Modeling Strategy

Use a hierarchical gray-box model:

```text
FuelFlow_total(x, M, tail) = FuelFlow_phys(x, M; theta_fleet, delta_tail) + Residual(x, M, tail)
```

where:

- `x` is the flight state
- `M` is Mach
- `theta_fleet` are fleet-level parameters shared across all tails
- `delta_tail` are tail-specific parameter offsets
- `Residual` is a small regularized correction term

This structure is preferred because it:

- respects known aircraft-performance relationships
- uses data efficiently across tails
- avoids overfitting sparse tail data
- gives interpretable tail-specific degradation/efficiency parameters
- supports direct optimization and uncertainty analysis

## Why Not RL, PDE, or PINN First

### RL

RL is not the best first method when the objective is pointwise MRC estimation for a given cruise state. MRC is fundamentally a one-dimensional constrained optimization problem once a performance model is available.

### PDE-first modeling

High-fidelity PDE-based aerodynamics is not a practical first choice from QAR alone. QAR does not provide the geometric, boundary-condition, or flow-field information needed for a CFD-first pipeline.

### PINN

A true PINN is not the best first choice here. PINNs are most useful when:

- the governing differential equations are central to the problem
- the equations can be imposed directly in the training loss
- the available data are sparse relative to the physics structure

In this use case, the relevant physics is lower-order flight-performance physics, not PDE reconstruction. A physics-informed or gray-box surrogate is a better fit than a true PINN.

## Inputs and Outputs

### Candidate Inputs

For each quasi-steady cruise segment, the useful variables are:

- pressure altitude or altitude
- gross weight
- total air temperature or ISA deviation
- Mach
- TAS, CAS, or equivalent derived airspeed measures
- wind or wind component
- fuel flow
- tail identifier
- engine bleed/anti-ice state if available
- route or atmospheric context if it materially changes fuel burn
- maintenance/degradation indicators if available

### Output

Primary output:

```text
MRC Mach for a given state and tail
```

Secondary outputs that should also be produced:

- predicted specific range curve versus Mach
- predicted fuel flow curve versus Mach
- uncertainty or confidence interval on MRC
- tail-level parameter estimates

## Data Preparation

The model should not be trained on arbitrary QAR rows. It should be trained on carefully filtered quasi-steady cruise segments.

### Recommended filtering

Keep rows or segments satisfying conditions close to:

- cruise phase only
- small vertical speed magnitude
- small acceleration magnitude
- no major turns
- stable flap/gear configuration
- no obvious sensor dropouts
- no extreme turbulence when building the baseline model

### Why segmentation matters

The steady-cruise mathematical formulation assumes approximate equilibrium:

- lift approximately equals weight
- thrust approximately equals drag
- speed changes slowly
- altitude changes slowly

If climb, descent, turns, or transient acceleration dominate, the simplified model becomes biased.

## Mathematical Formulation

### 1. Atmosphere model

At altitude `h`, use standard atmosphere relations with optional temperature deviation.

```text
T(h) = T_ISA(h) + DeltaT
rho(h) = p(h) / (R * T(h))
a(h) = sqrt(gamma * R * T(h))
```

where:

- `T` is local static temperature
- `rho` is air density
- `a` is speed of sound
- `gamma` is the ratio of specific heats
- `R` is the gas constant for air

### 2. Speed relations

For Mach `M`:

```text
TAS = M * a(h)
GS = TAS + WindComponent
q = 0.5 * rho * TAS^2
```

where `q` is dynamic pressure.

If only QAR airspeed channels are available, the exact TAS/CAS conversion should be handled consistently during preprocessing.

### 3. Steady cruise lift balance

For quasi-steady level cruise:

```text
L ~= W
```

Thus:

```text
CL = W / (q * S)
```

where:

- `W = m * g`
- `S` is wing reference area
- `CL` is lift coefficient

### 4. Drag model

Use a low-order drag polar:

```text
CD = CD0_tail + k_tail * CL^2 + CD_wave(M; Mcrit_tail, c_wave_tail)
```

with wave-drag rise modeled as:

```text
CD_wave(M) = c_wave_tail * max(0, M - Mcrit_tail)^2
```

Then drag is:

```text
D = q * S * CD
```

This captures:

- parasite drag through `CD0_tail`
- induced drag through `k_tail * CL^2`
- transonic drag rise through the wave-drag term

### 5. Steady cruise thrust requirement

For quasi-steady cruise:

```text
T_req ~= D
```

This avoids needing a detailed engine thrust model if the primary goal is fuel-flow estimation under steady conditions.

### 6. Fuel-flow model

Fuel flow is tied to required thrust through TSFC:

```text
FF_phys = TSFC_tail(h, DeltaT, z) * T_req
```

where `z` can include engine-operating context such as:

- bleed state
- anti-ice
- degradation state

A practical parameterization is:

```text
TSFC_tail(h, DeltaT, z) =
TSFC0_tail * (1 + alpha_h * phi_h(h)) * (1 + alpha_T * phi_T(DeltaT)) * (1 + alpha_z * phi_z(z))
```

The functions `phi_h`, `phi_T`, and `phi_z` can be simple normalized features.

### 7. Full gray-box performance model

The fleet-level physical model can be written as:

```text
FF_phys(x, M, tail) = D(x, M, tail) * TSFC_tail(x)
```

with tail-specific parameters:

```text
CD0_tail = CD0_fleet + delta_CD0_tail
k_tail = k_fleet + delta_k_tail
Mcrit_tail = Mcrit_fleet + delta_Mcrit_tail
c_wave_tail = c_wave_fleet + delta_cwave_tail
TSFC0_tail = TSFC0_fleet + delta_TSFC_tail
```

### 8. Residual correction model

Pure physics will not explain all operational effects, so add a small residual model:

```text
FF_total(x, M, tail) = FF_phys(x, M, tail) + r(x, M, tail)
```

The residual model `r` should be small and regularized. Suitable choices are:

- spline model
- Gaussian process on a reduced feature set
- gradient-boosted trees
- small neural network

The residual model should not replace the physics backbone. Its purpose is to correct systematic error, not to become the dominant model.

### 9. Specific range

Still-air specific range:

```text
SR(x, M, tail) = TAS(x, M) / FF_total(x, M, tail)
```

Ground-range objective:

```text
GR(x, M, tail) = GS(x, M) / FF_total(x, M, tail)
```

The optimization target must be aligned with the intended operational definition.

### 10. MRC optimization

For a feasible Mach interval:

```text
M in [M_min, M_max]
```

compute:

```text
MRC(x, tail) = argmax_M SR(x, M, tail)
```

or equivalently:

```text
MRC(x, tail) = argmin_M FF_total(x, M, tail) / TAS(x, M)
```

This is a one-dimensional bounded optimization problem. Suitable numerical methods:

- golden-section search
- Brent search
- dense grid plus local refinement

Because the decision variable is scalar, direct optimization is fast and stable.

## Hierarchical Tail-Specific Structure

A tail-specific model should not usually mean a completely separate unconstrained model per tail. The better structure is hierarchical.

### Fleet level

Shared across all tails:

- aerodynamic structure
- basic drag polar form
- altitude/temperature trends
- TSFC functional form

### Tail level

Estimated per tail:

- `delta_CD0_tail`
- `delta_TSFC_tail`
- `delta_Mcrit_tail`
- optional degradation indicator
- optional residual bias

This gives:

- data efficiency
- interpretable tail effects
- shrinkage toward fleet means when a tail has little data

Mathematically, tail effects can be treated as random effects or regularized fixed effects:

```text
delta_tail ~ N(0, Sigma_tail)
```

This is the core reason the model is scientifically stronger than training isolated per-tail black-box models.

## Estimation Objective

Suppose observed fuel flow is `FF_obs`. Then fit the model by minimizing:

```text
Loss =
sum_i w_i * [FF_obs_i - FF_total(x_i, M_i, tail_i)]^2
+ lambda_tail * ||delta_tail||^2
+ lambda_res * Omega(r)
```

where:

- `w_i` weights trusted steady-cruise samples more heavily
- `lambda_tail` regularizes tail-specific deviations
- `lambda_res` regularizes residual complexity
- `Omega(r)` is a residual smoothness or complexity penalty

If the target is fuel per nautical mile rather than fuel flow, the same structure can be used on:

```text
FPN = FF / GS
```

but fuel flow is usually the cleaner physical quantity to model before converting to range metrics.

## Practical Parameter Estimation Sequence

Recommended fitting sequence:

1. Fit the fleet-level physics parameters on all filtered cruise data.
2. Fit tail-specific offsets with regularization.
3. Diagnose systematic residuals.
4. Add a small residual correction model only if needed.
5. Validate MRC recommendations by replaying held-out QAR states.

This staged approach reduces overfitting and makes model failure modes easier to diagnose.

## Uncertainty Quantification

The model should provide uncertainty, not just a point estimate.

Recommended sources of uncertainty:

- parameter uncertainty from finite data
- residual model uncertainty
- measurement noise in QAR variables
- sensitivity of MRC to flat specific-range curves near the optimum

Useful outputs:

- confidence interval on predicted fuel flow curve
- confidence interval on MRC Mach
- confidence score that the optimum is well-defined

This matters because MRC can become numerically unstable when the specific-range curve is very flat around the optimum.

## Validation Strategy

Validation should be done at multiple levels.

### Fuel-model validation

On held-out tail-aware cruise data:

- MAE and RMSE of fuel flow
- MAE and RMSE of fuel per NM
- bias by altitude, weight, and temperature bins

### Tail-effect validation

Check whether estimated tail parameters are stable over time and physically plausible.

Examples:

- a higher-drag tail should consistently show worse efficiency
- a degraded tail should not randomly flip between very different parameter values

### MRC validation

Evaluate the derived optimum itself:

- compare predicted MRC against reconstructed local optimum from dense replay
- compare actual flown Mach to recommended Mach only as a behavioral reference, not as ground truth
- compare expected range improvement over operational baseline

## MRC vs LRC

It is important to distinguish:

- `MRC`: speed for absolute maximum range
- `LRC`: long-range cruise, usually slightly faster than MRC with only a small range penalty

Airlines often prefer LRC operationally because it trades a small fuel penalty for schedule robustness.

If the business objective is dispatch advice rather than pure aerodynamic optimum, it may be better to compute:

```text
LRC = fastest Mach such that SR(M) >= eta * SR(MRC)
```

for some threshold like:

```text
eta = 0.99
```

This can be added after the MRC model is built.

## Recommended Final Model

The recommended final model is:

1. Quasi-steady cruise data extraction from QAR.
2. Fleet-level gray-box aerodynamic and TSFC model.
3. Tail-specific parameter offsets estimated with hierarchical regularization.
4. Small residual correction model if needed.
5. Direct bounded optimization over Mach to produce MRC.

This is the best balance of:

- scientific defensibility
- interpretability
- robustness
- tail specificity
- data efficiency
- operational usefulness

## What Not To Do First

Avoid these as the primary first solution:

- PPO/RL for pointwise MRC estimation
- direct supervised prediction of flown Mach as if it were MRC
- a completely separate unconstrained neural network per tail
- a true PINN before establishing a strong gray-box baseline
- CFD/PDE-first modeling from QAR without appropriate aerodynamic inputs

## Implementation Roadmap

### Phase 1

- define steady-cruise filtering rules
- build a fleet-level fuel-flow model
- fit initial fleet parameters

### Phase 2

- add tail-specific offsets
- fit hierarchical regularization
- validate stability by tail

### Phase 3

- add residual correction model if systematic errors remain
- generate MRC and optional LRC recommendations
- quantify uncertainty

### Phase 4

- operationalize per-tail inference
- monitor drift and re-estimate tail parameters periodically

## Summary

The best modeling path for tail-specific MRC from QAR is a hierarchical gray-box mathematical model, not RL and not a PINN-first strategy.

The mathematics is based on:

- atmosphere relations
- Mach/TAS/GS conversion
- lift balance
- drag polar modeling
- thrust-drag balance
- TSFC-based fuel-flow estimation
- hierarchical parameter estimation
- bounded one-dimensional optimization

This gives a model that is scientifically meaningful, tail-specific, and directly aligned with the definition of MRC.
