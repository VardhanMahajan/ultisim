# Evolving strategies

How UltiSim turns "play yourself millions of times" into strategy discovery: the genomes, the algorithm, what a run writes, how to watch it, and what we've learned so far.

## Contents
1. [Why a genetic algorithm (and not AlphaZero)](#1-why-a-genetic-algorithm-and-not-alphazero)
2. [Genomes](#2-genomes)
3. [The algorithm](#3-the-algorithm)
4. [Running and watching](#4-running-and-watching)
5. [Run outputs](#5-run-outputs)
6. [Exploit audits: read this before believing a result](#6-exploit-audits-read-this-before-believing-a-result)
7. [Results so far](#7-results-so-far)

---

## 1. Why a genetic algorithm (and not AlphaZero)
AlphaZero's tree search works for turn-based, perfect-information board games. Ultimate has 14 agents moving at once in continuous space, with a lot of chance (throw error, drops). We keep AlphaZero's core loop (self-play, keep what wins, a learned evaluation) but search over **interpretable strategy genomes** with **coevolution**:

- **Self-play:** offenses and defenses evolve against each other.
- **Learned evaluation:** the thrower's completion model is fitted from self-play games (`scripts/fit_throw_model.py`), the way AlphaZero refits its value network.
- **League play:** a Hall of Fame of past champion defenses stops strategies from beating only the opponent of the moment (the AlphaStar trick).
- **Open-endedness:** novelty search and niche elites keep the search exploring instead of converging on one strategy.

Genomes are flat vectors of floats. The same vector is the GA's representation and the C engine's input struct.

## 2. Genomes
**Offense:** 6 formation slots (the non-throwers) plus team-level genes. Coordinates are in a force-aware frame: `dx` downfield of the disc, `dy` toward the open side. Angles: 0 = downfield, +90 = open side, 180 = back toward the disc.

**Defense:** a scheme (person / flex / zone), the mark, positioning, team-defense behaviours and 6 zone spots.

#### Offense genome (97 numbers)

| gene | size | range | meaning |
|---|---|---|---|
| `home_dx` | per slot (6) | -8 .. 35 | home spot, metres downfield of the disc |
| `home_dy` | per slot (6) | -16 .. 16 | home spot, metres toward the open side |
| `y_anchor` | per slot (6) | 0 .. 1 | 0 = home y follows the disc, 1 = home y is relative to the field centre |
| `handler` | per slot (6) | 0 .. 1 | > 0.5 = handler slot |
| `cutA_ang` | per slot (6) | -180 .. 180 | direction of cut option A (0 = downfield, +90 = open side, 180 = back) |
| `cutA_len` | per slot (6) | 3 .. 35 | length of cut A (m) |
| `cutB_ang` | per slot (6) | -180 .. 180 | direction of cut option B |
| `cutB_len` | per slot (6) | 3 .. 35 | length of cut B (m) |
| `cutA_bias` | per slot (6) | -3 .. 3 | logit preference for cut A over B |
| `read_def` | per slot (6) | 0 .. 3 | how strongly the cutter picks the option that moves away from their defender |
| `cut_delay` | per slot (6) | 0 .. 8 | seconds after the thrower is ready before this slot cuts |
| `recut_delay` | per slot (6) | 0.5 .. 6 | seconds after clearing before cutting again |
| `jab` | per slot (6) | 0 .. 1 | probability of a jab step before the cut |
| `reset_stall` | 1 | 2 .. 9 | stall count at which a handler comes for the reset |
| `reset_dx` | 1 | -8 .. 4 | reset spot, downfield of the disc |
| `reset_dy` | 1 | -12 .. 12 | reset spot, toward the open side |
| `give_go` | 1 | 0 .. 1 | chance the thrower cuts upfield after releasing |
| `risk_aversion` | 1 | 0 .. 3 | weight on the cost of a turnover |
| `huck_bias` | 1 | -0.3 .. 0.3 | extra utility for hucks |
| `break_bias` | 1 | -0.3 .. 0.3 | extra utility for break throws |
| `hammer_bias` | 1 | -0.3 .. 0.3 | extra utility for hammers |
| `patience` | 1 | 0.6 .. 1.6 | how much value the thrower places on holding for a better option |
| `panic_stall` | 1 | 6 .. 9.5 | stall count at which the best option is thrown regardless |
| `fake_rate` | 1 | 0 .. 1 | how often the thrower fakes when the mark takes away the best throw |
| `lead` | 1 | 0 .. 5 | metres to lead a moving receiver |
| `bid_aggr` | 1 | 0 .. 1 | chance to lay out when the disc is just out of reach |
| `seek_space` | per slot (6) | 0 .. 1 | bend cuts toward open space, away from nearby defenders |

#### Defense genome (32 numbers)

| gene | size | range | meaning |
|---|---|---|---|
| `zone` | 1 | 0 .. 1 | share of the 7 defenders in zone: k = round(zone*7); 0 person, 7 zone, 1-6 flex |
| `force` | 1 | 0 .. 5 | 0 forehand, 1 backhand, 2 straight up, 3 middle, 4 sideline |
| `mark_dist` | 1 | 0.9 .. 3 | metres between marker and thrower (closer = wider coverage, less reaction time) |
| `mark_angle` | 1 | 0 .. 110 | marker's position, degrees from downfield toward the break side |
| `cushion_base` | 1 | 0.3 .. 5 | metres of cushion on your attacker |
| `cushion_per_m` | 1 | 0 .. 0.25 | extra cushion per metre the attacker is from the disc |
| `angle_near` | 1 | -180 .. 180 | positioning angle vs. attackers near the disc (0 = between attacker and disc, 180 = deep, +90 = open side) |
| `angle_far` | 1 | -180 .. 180 | same, for attackers far from the disc |
| `anticipation` | 1 | 0 .. 0.6 | how much defenders extrapolate their (delayed) read |
| `poach` | 1 | 0 .. 1 | sag into the lanes off stationary attackers |
| `help_margin` | 1 | -0.3 .. 1 | seconds of slack when deciding to go for a disc in the air |
| `bid_aggr` | 1 | 0 .. 1 | chance to lay out for a block |
| `match_speed` | 1 | 0 .. 2 | matchup cost on speed mismatch (fast on fast) |
| `zone_dx` | per slot (6) | -6 .. 40 | zone spots 1-6, metres downfield of the disc |
| `zone_dy` | per slot (6) | -18 .. 18 | zone spots 1-6, metres toward the open side |
| `zone_shade` | 1 | 0 .. 1 | how far zone players shade toward the nearest attacker |
| `switch` | 1 | 0 .. 1 | 0 never switch, 1 always re-match to the nearest attacker (clam) |
| `help_cut` | 1 | 0 .. 1 | how far a free defender steps into the lane of a cutter who beat their man |
| `stack_sag` | 1 | 0 .. 1 | defenders of stationary attackers in a cluster sag into the stack |
| `flex_focus` | 1 | 0 .. 1 | flex: person defenders take the deepest (1) or closest-to-disc (0) cutters |
| `transition_throws` | 1 | 0 .. 10 | zone/flex switches to person D after this many completions (>= 9.5 never) |
| `trap` | 1 | 0 .. 1 | near a sideline the short zone players collapse on the thrower |

Seed strategies ([`ultisim/genome.py`](../ultisim/genome.py)):

| Offense seeds | `vertical`, `horizontal`, `side`, `iso`, `zone_o` |
|---|---|
| **Defense seeds** | `person_forehand`, `person_backhand`, `person_straight_up`, `person_middle`, `person_sideline`, `cup_zone`, `wall_zone`, `clam`, `help_d`, `flex_deep`, `junk_cup`, `zone_to_person`, `cup_trap` |

**Structure families** ([`ultisim/classify.py`](../ultisim/classify.py)) are read from the cutter slots' home spots:
- **vertical:** strung out along the field, roughly central
- **side:** strung out along the field, off to one side
- **horizontal:** spread across the field at a similar depth
- **iso:** one cutter isolated in front of a clump
- **spread:** no dominant axis
- **other:** fewer than 2 cutters.

Defense schemes are labelled `person_<force>`, `flex_<k>z<7-k>p` (k zone players) or `zone`.

## 3. The algorithm
[`ultisim/evolve.py`](../ultisim/evolve.py), run with [`scripts/run_offense.py`](../scripts/run_offense.py). Each generation:

1. **Fresh rosters.** Sample `n_rosters` roster pairs, shared by every genome this generation (common random numbers: differences come from strategy, not luck of the draw).
2. **Opponents.** Every offense plays 6 opponents: 3 from the defense population (balanced round-robin), 1 from the Hall of Fame, 2 hand-built defenses.
3. **Racing.** Everyone plays `stage1` possessions (default 96). The top `race_frac` (34%) play on to `stage2` (288), so weak genomes don't waste simulation time.
4. **Fitness.** Offense = score rate per possession. Defense = 1 − score rate conceded.
5. **Behaviour fingerprints.**
   - Offense: huck/break/hammer share, throws per possession, possession length, and formation shape (depth, spreads, lateral offset, handler count).
   - Defense: its scheme genes.
6. **Novelty.** Mean distance to the 10 nearest fingerprints in the population plus an archive of past behaviours (up to 600).
7. **Breeding.**
   - Elitism (top 4).
   - **Niche elites:** the best member of every structure family (offense) and scheme family (defense) always survives.
   - Tournament selection on `rank(score) + 0.35 × rank(novelty)`.
   - Slot-block crossover: a whole formation slot comes from one parent.
   - Gaussian mutation: 15% of genes, σ = 8% of each gene's range.
   - 6% chance of a **macro-mutation** that redraws one entire slot.
   - **3 immigrants** per generation: random genomes, or heavily reshaped survivors.
8. **Hall of Fame.** The generation's best defense joins (last 20 kept).
9. **Logging, checkpoint, replay.** Write `history.jsonl`, `status.json`, `archive.json` and `state.pkl`. Every `replay_every` generations, render a champion replay in the background with the Python engine.

Important options (see `--help`):

| option | default | meaning |
|---|---|---|
| `--gens` | 100 | generations |
| `--off-pop`, `--def-pop` | 48, 32 | population sizes |
| `--stage1`, `--stage2`, `--race-frac` | 96, 288, 0.34 | racing budget |
| `--rosters` | 64 | roster pairs per generation |
| `--replay-every` | 2 | champion replay frequency |
| `--seed` | 0 | run seed |
| `--resume` | | continue from `runs/<run>/state.pkl` |

The novelty weight, niche elites, immigrants and macro-mutation rate are `EvoConfig` fields in `ultisim/evolve.py`.

**Cost:** games per generation ≈ `off_pop × stage1 + race_frac × off_pop × (stage2 − stage1)`, about 7,700 at the defaults. Divide by your `scripts/bench.py` rate to get seconds per generation.

## 4. Running and watching
```bash
python scripts/run_offense.py --run myrun --gens 100            # Ctrl+C saves a checkpoint
python scripts/run_offense.py --run myrun --gens 300 --resume     # continue
python viz/server.py                                              # http://127.0.0.1:8765/dashboard.html?run=myrun
```

The **dashboard** polls the run every 4 s and shows:
- offense best/median/mean and defense best per generation
- the structure-family mix over time
- the best score rate ever reached by each family
- the defense scheme mix
- champion offense and defense genes
- play stats
- an embedded live replay of the current champion.

The **replay viewer** (`index.html?replay=/runs/<run>/latest_best.json&live=1`) plays any replay with scrubbing, speed control, throw arcs, the mark wedge and the stall count.

## 5. Run outputs
Everything goes into `runs/<run>/` (formats: [`viz/RUN_FORMAT.md`](../viz/RUN_FORMAT.md), [`viz/REPLAY_FORMAT.md`](../viz/REPLAY_FORMAT.md)):

| file | content |
|---|---|
| `meta.json` | run settings |
| `history.jsonl` | one line per generation: fitness, family/scheme counts, champions, stats, novelty |
| `status.json` | live state (`running` / `finished` / `stopped` / `error`) |
| `archive.json` | best offense ever per structure family |
| `latest_best.json` | champion replay |
| `state.pkl` | full checkpoint: populations, Hall of Fame, archives, RNG |

## 6. Exploit audits: read this before believing a result
Evolution optimizes the **simulator**, not real Ultimate. Any shortcut in the physics gets found and exploited, usually within 10–30 generations. Red flags:

- One family takes over the whole population very fast.
- A defense concedes far less (or an offense scores far more) than every hand-built strategy.
- A gene sits pinned at the edge of its range.
- Play statistics drift far from the calibration bands (break share, hammer share, completion rate).

Tools:
- `scripts/audit_defenses.py --run <run>` re-evaluates a run's defenses against the hand-built offenses with many possessions and prints the strongest genomes in full.
- Watch the champion replay.
- Re-run the calibration with the champion's genes.

Exploits found so far (all fixed, with regression tests or calibration checks):

| Run | What evolution found | Root cause | Fix |
|---|---|---|---|
| smoke | Defenses conceding 9–12% with a 140° mark wedge | Mark width had no cost | Fixed blocking capacity, then a body-based mark |
| watch1 | 31 of 32 defenses = straight-up mark at max width | Wide coverage of every forward throw | Body-based mark: coverage from mark distance; tight marks react later |
| watch2 | The hand-built zone offense threw hammers on 8–27% of throws | Hammers went over the mark and were untouchable, with no cost | Hammer catch penalty and ×2 error |
| watch3 | Offenses breaking the mark on 45–92% of throws at 90–97% completion | Narrow mark cone; curved around-breaks judged only by their first 2 m | Break-side tail, release/target bisector, more curve error |
| watch4 | `flex_1z6p` defenses conceding 3–4% | **Fronting** (`angle_near` ≈ −4°, cushion 1.9 m + 0.22 m per m from the disc). In flight, person defenders positioned against the live disc, so a fronting defender slid onto every lane for free. The thrower's read assumed defenders drift on their current velocity, so it rated these throws 0.91 when they completed 0.76. Not the flex or switching: switching off changed nothing | In-flight read delay for all positional defenders; the thrower predicts the receiver's defender holding their spot; blind-side delay; throw model fitted on mutated and evolved strategies too. The champion now concedes 0.12, and evolved offenses reach 0.33 against it (regression test `test_thrower_sees_fronting_defender_step_into_the_lane`) |

## 7. Results so far
These runs were short (100 generations, about 10 minutes). Treat them as tests of the machinery.

**watch3** (novelty search on; before the break fix):
- **Diversity:** every structure family stayed alive for 100 generations. Champion family by generation: iso 39, vertical 24, side 16, spread 14, horizontal 6.
- **Arms race:** the average offense fell from 0.42 to 0.34 score rate, and the average defense's conceded rate fell from 0.43 to 0.28.
- **Emergent defense:** person D forcing sideline, marker sitting back at 3 m, defenders fronting their attackers with cushion growing with distance from the disc, constant switching, always bidding. It conceded 11–19% to the hand-built offenses; hand-built defenses allow 30–55%.
- **Emergent offense:** a 3–4-handler pod near the disc, a mid cutter about 20 m deep and a deep cutter about 35 m deep. Its edge came mostly from the break exploit (now fixed); after the fix it is level with the hand-built stacks.

**watch4** (first run on the break-fixed simulator; 100 generations, 20 min). Exploit found, **fixed since** (see the table above):
- Defenses won decisively. The average offense score fell from 0.34 to 0.30, the average conceded rate from 0.32 to 0.22, and the best defenses reached 0.05–0.10.
- By generation 99, 28 of 32 defenses were `flex_1z6p`: one dedicated marker who sprints to every new thrower, with six person defenders.
- The audit shows these concede **only 2.7–4.4%** to the hand-built offenses. That's far beyond anything plausible.
- **First suspicion (wrong):** free instant re-matching after every catch and every switch. Turning switching off, or playing the same genes as plain person D, changed the conceded rate by less than 0.01.
- **Actual cause:** the genes that mattered were the positioning ones. Resetting `angle_near`, `angle_far` or `mark_angle` to standard values each raised the conceded rate to 0.15–0.18. The defenders fronted. 90% of the Ds were by the receiver's own defender, who stood about halfway along a short throw at release.
  - In flight, person defenders positioned against the live disc, sliding onto the lane with no read.
  - The thrower's read assumed a defender who hasn't read the throw keeps drifting on their current velocity, rather than holding their spot next to the receiver.
- **After the fix:** the champion concedes 0.12 to the hand-built offenses (0.048 before). Offenses evolved against it alone (32 offenses, 30 generations) reached 0.33 by throwing more aggressively and deeper, and still score 0.34–0.39 against person D.
- Hammers also crept back to 14% of throws by the end, as the offenses' release valve against these defenses. Re-check after the fix.
- Champion family by generation: spread 42, vertical 16, side 15, horizontal 14, iso 13. Best score ever per family: vertical 0.54, iso 0.54, spread 0.52, horizontal 0.51, side 0.47.

The questions the project set out to answer (best offensive structure with random-stat teams; best defense against vertical and horizontal stacks) need long runs plus a final tournament of each family's champion. See the roadmap in [PLAN.md](PLAN.md), Phases 4–6.
