# The simulation model

This document describes how UltiSim plays a point of Ultimate. Every number mentioned here lives in [`configs/default.toml`](../configs/default.toml), which is commented line by line. Change a value there and both engines pick it up.

The Python reference engine ([`ultisim/engine_ref.py`](../ultisim/engine_ref.py)) is the readable source of truth. The C engine ([`csim/sim.c`](../csim/sim.c)) is a function-by-function port that must match it statistically (see [ARCHITECTURE.md](ARCHITECTURE.md)).

## Contents
1. [Field, rules and modes](#1-field-rules-and-modes)
2. [Players](#2-players)
3. [Movement](#3-movement)
4. [Throws and disc flight](#4-throws-and-disc-flight)
5. [How the thrower decides](#5-how-the-thrower-decides)
6. [The mark](#6-the-mark)
7. [When a D happens](#7-when-a-d-happens)
8. [Offensive behaviour](#8-offensive-behaviour)
9. [Defensive schemes](#9-defensive-schemes)
10. [Calibration](#10-calibration)

---

## 1. Field, rules and modes
- **Field:** WFDF, 100 × 37 m (64 m playing proper + two 18 m end zones). Units are metres and seconds.
- **Frame:** internally the current offense always attacks +x. On a turnover the state is rotated 180°.
- **Tick:** 0.1 s. Each tick runs, in order: thrower decision, behaviour targets for all 14 players, movement, the disc in flight, then the stall count.
- **Stall:** counts at 1 per second only while the marker is within 3 m of the thrower. It's a turnover at 10.
- **Double team:** only the marker may stand within 3 m of the thrower, unless another defender is guarding an attacker who is also inside that radius. Nobody but the marker can touch the disc within 3 m of the release.
- **Possession mode** (used for training): the offense starts with the disc on its own goal line at a random width, both teams set, and the possession runs until a score or turnover.
- **Point mode** (used for validation): turnovers switch possession at the turnover spot (moved to the goal line if it's in an end zone) until someone scores.
- **Not modelled:** wind, fouls and picks, pulls, timeouts, fatigue, travels. There's a 120 s possession cap that counts as a turnover.

## 2. Players
Each roster has 7 players with stats drawn from club-level, open-division distributions. Stats are correlated through two hidden factors:
- an **athletic** factor drives speed, acceleration, vertical and agility
- a **handler** factor drives throwing skill, break skill, ranges, release time and decision quality.

| Stat | Default distribution | Used for |
|---|---|---|
| top speed | N(7.5, 0.6) m/s, 6.0 to 9.2 | running |
| acceleration | N(4.0, 0.5) m/s² | running; braking is stronger, scaled by agility |
| agility | U(0, 1) | braking power, so cuts cost less time |
| reaction time | N(0.27, 0.04) s | how late a defender reacts to their attacker (the cutter's head start) |
| height, vertical | N(1.80, 0.07) m, U(0.4, 0.8) m | catch/block radius, contested catches |
| hands | U(0.980, 0.998) | uncontested catch probability |
| throw skill | U(0, 1) | throw error |
| break skill | U(0, 1) | beating the mark |
| decision noise | U(0, 1) | how badly the thrower misjudges completion odds |
| marking | U(0, 1) | block chance at the mark, recovery from fakes |
| release time | U(0.30, 0.60) s | catch to earliest release |
| ranges | BH 45–70, FH 35–60, hammer 15–40 m | longest throw of each type |

Archetype rosters (`fast`, `throwing`, `tall`, ...) shift the hidden factors: see [`ultisim/roster.py`](../ultisim/roster.py).

## 3. Movement
- **Point masses.** Players accelerate at `accel` and brake at `accel × (1.2 + 0.8 × agility)`, so a sharp cut means braking away your old velocity first. Desired speed follows an "arrive" profile, so players stop on their target instead of overshooting.
- **Defender reaction.** Defenders track a **delayed** copy of their attacker: position and velocity as they were `reaction_time` ago. They match that delayed velocity and correct their position toward their scheme spot. The reaction delay is what lets cuts gain separation.
- **Catch radius.** Reach is about 0.9 m plus height and vertical terms; the 0.9 m includes a lunge step. A layout bid adds 1.5 m but succeeds less often and puts the player on the ground for 1.5 s.

## 4. Throws and disc flight
- **Types:**
  - backhand and forehand, each straight, inside-out (+1 curve) or outside-in (−1)
  - hammer
  - any backhand or forehand longer than 32 m flies as a **huck**.
- **Flight path.** A quadratic Bezier from the release point (1 m to the throwing-arm side) to the landing point. Flight time = hang + distance / speed: bullets about 18 m/s, hucks 17.5 m/s with 0.45 s hang, hammers 12.5 m/s with 0.6 s hang.
- **Tail.** After reaching its aim point the disc glides on for up to 0.35 s along its tangent before landing.
- **Error.** The landing point = aim + Gaussian error with σ = (0.20 + 0.030·d + 0.0018·d²) × skill (×0.6 best to ×1.3 worst) × type multiplier (hammer ×2.0), with further multipliers for break throws ×1.15, a set mark ×1.10, and curved throws ×1.25.
- **Height profile.** Every throw has a height curve: release (1.0 m, or 2.2 m overhead for hammers), a peak at mid-flight, then catch height (1.2 m). The tail sinks to 0.3 m.

  | throw | peak height |
  |---|---|
  | flat bullet | 1.0 + 0.025·d (about 1.2–2 m) |
  | lofted touch pass | 2.0 + 0.05·d, capped at 4.5 m |
  | huck | 2.5 + 0.09·d (35 m → 5.7 m) |
  | hammer | 6 + 0.05·d |
- **Flat vs lofted.** Every straight backhand or forehand between 6 and 35 m can also be thrown **lofted**: 25% slower with 0.35 s extra hang. It goes over a defender in the lane, but everyone gets more time to close.
- **Reach envelope.**
  - Standing reach = 1.25 × height (about 2.0–2.5 m).
  - Jumping adds the player's vertical, but a jumping player's lateral reach radius drops to ×0.6.
  - A disc above someone's reach envelope can't be touched by them.
  - Discs below 0.4 m are hard for anyone: catch and block chance ×0.6.
  - Layout bids only happen for discs below standing reach.
- **Huck read delay.** Everyone reads hucks 0.3 s later.

## 5. How the thrower decides
Every `decision_interval` (0.4 s) after the release time, the thrower builds candidate throws:
- 6 receivers × 2 leads (on the receiver, or `lead` metres in front)
- × throw types: straight BH/FH (flat, plus a lofted version between 6 and 35 m) always; curves and hammers only toward **break-side** targets, since their job is beating the mark.

For each candidate it estimates the completion probability `p`:

1. **Analytic part.** For every defender and 10 points along the flight:
   - **Arrival time.** A closed-form time-to-reach that accounts for current velocity, turning cost, reaction delay and drift.
   - **Reach envelope.** At each path point the defender's reach depends on the disc's height there: out of reach above their envelope, smaller radius when they'd have to jump.
   - **Throw error.** The defender's reach is widened by the throw's error, since the real flight wanders.
   - **Receiver's own defender.** Modelled as shadowing the receiver to the catch point.
   - **Combined with:** the mark block chance, receiver arrival, error vs adjust radius, out-of-bounds risk and hands.
2. **Learned correction.** A logistic regression on 23 features of that analysis (margins of the most dangerous defender, distance, flight time, huck/hammer/curve/break/lofted flags, stall, receiver speed, ...), fitted from self-play with exploration by [`scripts/fit_throw_model.py`](../scripts/fit_throw_model.py). It's calibrated to within a few points in every probability bucket. **Refit it whenever the physics change.**

The thrower then perceives `p̂ = sigmoid(logit p + noise)`, with noise growing with `decision_noise`, and scores each candidate:

```
U = p̂ · V(target) − (1 − p̂) · risk_aversion · C(target) + huck/break/hammer biases
```

- `V` is the value of having the disc at a spot: `0.45 + 0.45·u^1.5` along the field, minus a sideline penalty, and 1.0 in the end zone.
- `C` is the opponent's value from the mirrored spot.

It throws the best candidate if `U` beats the value of holding: `V(here) · patience · (1 − (s/10)²)`, where `s` is the stall count or the time held, whichever is larger. Past `panic_stall` it throws regardless.

**Possession urgency.** After 40 s of possession, patience and risk aversion fade linearly to 40% over the next 40 s. Nobody swings the disc forever; without this, stack offenses facing a cup zone could hit the time cap on 40%+ of possessions. If the best option is a break the mark is taking away, it may fake first, which shifts the marker's coverage for about 0.35 s.

## 6. The mark
A set marker (within 3 m) blocks a throw with probability

```
mark_block_max (0.55) × [ spread × cover + tail ] × skill × type × fake × quality
```

- **Body cone.** `cover = 1 − (Δ/w)²` inside a cone of half-width `w = atan(1.25 m / mark_dist)`. Δ is the angle between the throw and the marker's direction from the thrower. A tight 0.9 m mark covers about ±54°, a 2 m mark about ±32°.
- **Break-side tail.** `tail = 0.6 × (1 − Δ/65°) × (1 − cover)`. The marker shuffles and reaches for anything released toward their side. It fades out well before the open side.
- **Which direction counts.** A throw is judged along the **bisector of its release direction and its target direction**, so a curve can bend a break around the mark but can't hide it completely.
- **Quality** = `1 − exp(−mark_dist / 0.8)`: a tight mark has less time to react to the release.
- **Type:** hammers ×0.15, because they go over the mark. **Fake:** × (0.35 + 0.4 × marking) during the fake window.
- **Break:** a throw counts as a break if it's aimed within 50° of the marker.

## 7. When a D happens
- **Mark block:** at release, with the probability above.
- **Interception / block:** a defender touches the disc in flight. A swept test in each player's own moving frame means nothing tunnels through between ticks. It's a block with probability 0.88.
- **Reflex chance:** a defender the disc reaches before they've read the throw (reaction + read delay) only gets 0.35 of that.
- **Help defenders read later:** person-D help defenders (not guarding the receiver) read throws 0.35 s later, because they're watching their own player. Zone players face the disc and don't pay this.
- **Contested catch:** when the receiver and a defender touch in the same tick, a logistic contest on effective height (height + 0.6 × vertical) and who's closer decides it.
- **Drop:** catch probability = hands − speed term − 0.015 if contested − 0.12 for hucks (over the shoulder) − 0.10 for hammers.
- **Throwaway / out of bounds:** the disc lands untouched, or is caught outside the field.
- **Stall:** the count reaches 10.

## 8. Offensive behaviour
Six formation **slots** (the non-throwers) each have:
- a home spot relative to the disc, in a force-aware frame (dx downfield, dy toward the open side), optionally anchored to the field centre
- a handler flag
- two cut options (direction and length), chosen by a logit that prefers the option moving away from the cutter's own defender (`read_def`)
- a cut delay, a re-cut delay, a jab-step probability
- `seek_space`, which bends the cut toward open space.

Players are **re-matched to slots after every throw** (Hungarian assignment, with handlers preferred for handler slots), so roles flow the way they do in a real stack. A handler cuts for the reset at `reset_stall`. After releasing, the thrower may give-and-go.

## 9. Defensive schemes
The `zone` gene sets how many of the 7 defenders play zone: `k = round(zone × 7)`.

- **Person D (k = 0):** matchups by speed and distance. Defenders position relative to their attacker with an angle (0 = between the attacker and the disc, 180 = deep side, +90 = open side) that blends from `angle_near` to `angle_far` with distance from the disc. Cushion = `cushion_base + cushion_per_m × distance`.
  - `poach`: sag into the lanes off stationary attackers.
  - `switch` (clam): every 0.5 s defenders may re-match to the nearest attackers if that saves enough running.
  - `help_cut`: when a cutter beats their defender by more than 2.5 m, the nearest free defender steps into that cutter's lane.
  - `stack_sag`: defenders of stationary attackers in a cluster sag into the stack.
- **Zone (k = 7):** a marker plus 6 zone spots relative to the disc, re-assigned after every catch. Zone players shade toward attackers in their area (`zone_shade`) and chase throws.
  - **Read delay:** zone spots follow where the disc was 0.45 s ago (reaction + read), so a quick swing can beat the cup.
  - `trap`: near a sideline the short zone players collapse on the thrower.
- **Flex (k = 1..6):** the marker and k−1 zone spots, with person D on the 7−k cutters chosen by `flex_focus` (deepest first, or closest to the disc first).
  - `transition_throws`: a zone or flex switches to person D after N completions.
- **Force:** forehand, backhand, straight up, middle or sideline. It sets the open side and the marker's side.

The seed defenses are: `person_<force>` ×5, `cup_zone`, `wall_zone`, `clam`, `help_d`, `flex_deep`, `junk_cup`, `zone_to_person` and `cup_trap`.

## 10. Calibration
[`scripts/calibrate.py`](../scripts/calibrate.py) plays the hand-built offenses against the hand-built defenses and compares the results with club-level target bands. Use `--engine c` for a 30-second run.

| Metric | Target band | Current |
|---|---|---|
| completion rate | 0.90–0.95 | 0.92 |
| score rate per possession (from the goal line) | 0.40–0.60 | **0.39** (slightly low) |
| hold rate (full points) | 0.55–0.72 | **0.54** (slightly low) |
| throws per scoring possession | 8–15 | 9.3 |
| break share (vs person D) / completion | ~10–20% / ~0.80 | ~10% / 0.82 |
| huck share / completion | 4–15% / 0.45–0.62 | 3.6% / 0.66 |
| D share of turnovers | 25–50% | 52% |

**Known deviations:**
- Scoring and hold rate sit about 3% under their bands, and Ds are slightly over. Lowering `block_prob` doesn't help, because throwers adapt; the limit is the risk/value trade-off across about 9 throws per possession.
- Hucks are rarer than in real play (throwers huck only when it's open) and slightly too easy.
- Turnovers lean toward drops (~30%) over throwaways (~18%).
- There are no stall-outs; possession urgency makes throwers force throws before that.
- Hammers are almost never chosen. The catch penalty, doubled error and height model make them rarely worth it; a real hammer's niche (over a cup to a popper) may need a dedicated look.

**Sensitivity:** ±20% on any key constant moves results smoothly. Throw accuracy is the strongest lever; defender reaction time is surprisingly weak. The [development log](PLAN.md) records every calibration step with numbers.
