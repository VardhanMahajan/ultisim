"""ctypes bindings to the C core (build/ultisim_core.*) and the single source of truth for its data layout.

The C struct definitions in csim/us_structs.h are GENERATED from the field lists below by
scripts/build_c.py, so Python and C can never disagree about a layout. Bump ABI_VERSION whenever a
list changes (the build writes it into the header too).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

from .genome import DEFENSE_SCHEMA, OFFENSE_SCHEMA
from .roster import Roster

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
ABI_VERSION = 11

N_MODEL_W = 23  # learned throw-model weights (len(engine_ref.FEATURE_NAMES))

# (C field name, dotted path into the TOML config). Every field is a double.
CONFIG_FIELDS: list[tuple[str, str]] = [
    ("L", "field.length"), ("W", "field.width"), ("ez", "field.endzone_depth"),
    ("dt", "sim.dt"), ("max_poss_time", "sim.max_possession_time"), ("max_point_poss", "sim.max_point_possessions"),
    ("decision_interval", "sim.decision_interval"), ("urgency_start", "sim.urgency_start"),
    ("urgency_span", "sim.urgency_span"), ("urgency_floor", "sim.urgency_floor"), ("explore_eps", "sim.explore_eps"),
    ("start_x", "sim.start_x"), ("start_y_margin", "sim.start_y_margin"),
    ("stall_max", "rules.stall_max"), ("stall_rate", "rules.stall_rate"), ("pickup_grace", "rules.pickup_grace"),
    ("double_team_radius", "rules.double_team_radius"),
    ("jog_fraction", "movement.jog_fraction"), ("brake_base", "movement.brake_base"),
    ("brake_agility", "movement.brake_agility"), ("reach_base", "movement.reach_base"),
    ("reach_height", "movement.reach_height"), ("reach_vert", "movement.reach_vert"),
    ("layout_extension", "movement.layout_extension"), ("layout_ground_time", "movement.layout_ground_time"),
    ("layout_success_scale", "movement.layout_success_scale"), ("disc_read_delay", "movement.disc_read_delay"),
    ("huck_read_extra", "movement.huck_read_extra"), ("boundary_slack", "movement.boundary_slack"),
    ("err_a", "throws.err_a"), ("err_b", "throws.err_b"), ("err_c", "throws.err_c"),
    ("skill_err_best", "throws.skill_err_best"), ("skill_err_worst", "throws.skill_err_worst"),
    ("break_err_mult", "throws.break_err_mult"), ("marked_err_mult", "throws.marked_err_mult"),
    ("curve_err_mult", "throws.curve_err_mult"), ("huck_min_dist", "throws.huck_min_dist"),
    ("min_throw_dist", "throws.min_throw_dist"), ("tail_time", "throws.tail_time"),
    ("model_error_margin", "throws.model_error_margin"), ("read_noise_base", "throws.read_noise_base"),
    ("read_noise_skill", "throws.read_noise_skill"), ("receiver_adjust", "throws.receiver_adjust"),
    ("fake_time", "throws.fake_time"), ("fake_effect_time", "throws.fake_effect_time"),
    ("fake_shift_deg", "throws.fake_shift_deg"), ("specials_everywhere", "throws.specials_everywhere"),
    ("bh_speed", "throws.backhand.speed"), ("bh_hang", "throws.backhand.hang"),
    ("bh_huck_speed", "throws.backhand.huck_speed"), ("bh_huck_hang", "throws.backhand.huck_hang"),
    ("bh_release_side", "throws.backhand.release_side"), ("bh_mark_mult", "throws.backhand.mark_mult"),
    ("bh_err_mult", "throws.backhand.err_mult"),
    ("fh_speed", "throws.forehand.speed"), ("fh_hang", "throws.forehand.hang"),
    ("fh_huck_speed", "throws.forehand.huck_speed"), ("fh_huck_hang", "throws.forehand.huck_hang"),
    ("fh_release_side", "throws.forehand.release_side"), ("fh_mark_mult", "throws.forehand.mark_mult"),
    ("fh_err_mult", "throws.forehand.err_mult"),
    ("ham_speed", "throws.hammer.speed"), ("ham_hang", "throws.hammer.hang"),
    ("ham_release_side", "throws.hammer.release_side"), ("ham_mark_mult", "throws.hammer.mark_mult"),
    ("ham_err_mult", "throws.hammer.err_mult"),
    ("loft_speed_mult", "throws.loft_speed_mult"), ("loft_hang_extra", "throws.loft_hang_extra"),
    ("loft_min_dist", "throws.loft_min_dist"), ("loft_max_dist", "throws.loft_max_dist"),
    ("h_release", "throws.height.release"), ("hammer_release", "throws.height.hammer_release"), ("h_catch", "throws.height.catch"), ("h_ground", "throws.height.ground"),
    ("flat_hbase", "throws.height.flat_base"), ("flat_hper_m", "throws.height.flat_per_m"),
    ("loft_hbase", "throws.height.loft_base"), ("loft_hper_m", "throws.height.loft_per_m"), ("loft_hmax", "throws.height.loft_max"),
    ("huck_hbase", "throws.height.huck_base"), ("huck_hper_m", "throws.height.huck_per_m"),
    ("hammer_hbase", "throws.height.hammer_base"), ("hammer_hper_m", "throws.height.hammer_per_m"),
    ("low_disc", "throws.height.low_disc"), ("low_mult", "throws.height.low_mult"),
    ("standing_reach_ratio", "movement.standing_reach_ratio"), ("jump_reach_mult", "movement.jump_reach_mult"),
    ("curve_amount", "throws.curve.amount"),
    ("mark_block_max", "defense.mark_block_max"), ("mark_skill_weight", "defense.mark_skill_weight"),
    ("break_skill_weight", "defense.break_skill_weight"), ("mark_set_radius", "defense.mark_set_radius"), ("mark_width_ref", "defense.mark_width_ref"),
    ("mark_reach", "defense.mark_reach"), ("mark_break_tail", "defense.mark_break_tail"), ("mark_tail_deg", "defense.mark_tail_deg"),
    ("break_side_deg", "defense.break_side_deg"), ("mark_react_dist", "defense.mark_react_dist"),
    ("switch_interval", "defense.switch_interval"), ("switch_gain", "defense.switch_gain"),
    ("help_sep", "defense.help_sep"), ("zone_read_delay", "defense.zone_read_delay"), ("stack_radius", "defense.stack_radius"), ("trap_zone", "defense.trap_zone"),
    ("block_prob", "defense.block_prob"), ("contest_base", "defense.contest_base"),
    ("contest_height_weight", "defense.contest_height_weight"),
    ("contest_position_weight", "defense.contest_position_weight"), ("reflex_block", "defense.reflex_block"),
    ("help_read_delay", "defense.help_read_delay"),
    ("speed_drop_per_ms", "catching.speed_drop_per_ms"), ("contest_drop", "catching.contest_drop"),
    ("huck_drop", "catching.huck_drop"), ("hammer_drop", "catching.hammer_drop"),
    ("v_base", "value.base"), ("v_gain", "value.gain"), ("v_power", "value.power"), ("v_edge", "value.edge_penalty"),
    ("use_model", "<computed>"),
] + [(f"model_w{i}", "<computed>") for i in range(N_MODEL_W)]

PLAYER_FIELDS = Roster.stat_names()                     # 15 doubles, in Roster order
OFFENSE_FIELDS = [(n, k) for n, k, _, _ in OFFENSE_SCHEMA]  # GA flat vector == C struct
DEFENSE_FIELDS = [(n, k) for n, k, _, _ in DEFENSE_SCHEMA]

OUTCOMES = ("score", "block", "interception", "drop", "throwaway", "out_of_bounds", "stall", "mark_block", "timeout")

# Result: int32 fields then double fields.
RESULT_INT_FIELDS = ["outcome", "offense", "n_poss", "scorer", "throws", "completions", "hucks", "huck_completions",
                     "breaks", "break_completions", "hammers", "fakes"] + [f"n_{o}" for o in OUTCOMES]
RESULT_DBL_FIELDS = ["duration", "max_stall", "end_x", "end_y", "sum_stall_at_throw", "sum_p_chosen"]


def _struct(name, fields):
    return type(name, (ctypes.Structure,), {"_fields_": fields})


UsPlayer = _struct("UsPlayer", [(n, ctypes.c_double) for n in PLAYER_FIELDS])
UsConfig = _struct("UsConfig", [(n, ctypes.c_double) for n, _ in CONFIG_FIELDS])
UsOffense = _struct("UsOffense", [(n, ctypes.c_double * k) if k > 1 else (n, ctypes.c_double) for n, k in OFFENSE_FIELDS])
UsDefense = _struct("UsDefense", [(n, ctypes.c_double * k) if k > 1 else (n, ctypes.c_double) for n, k in DEFENSE_FIELDS])
UsGameSpec = _struct("UsGameSpec", [
    ("players", UsPlayer * 14), ("off_idx", ctypes.c_int32 * 2), ("def_idx", ctypes.c_int32 * 2),
    ("mode", ctypes.c_int32), ("_pad", ctypes.c_int32), ("seed", ctypes.c_uint64),
    ("start_x", ctypes.c_double), ("start_y", ctypes.c_double),
])
UsResult = _struct("UsResult", [(n, ctypes.c_int32) for n in RESULT_INT_FIELDS]
                   + ([("_pad", ctypes.c_int32)] if len(RESULT_INT_FIELDS) % 2 else [])
                   + [(n, ctypes.c_double) for n in RESULT_DBL_FIELDS])


C_RESERVED = {"switch": "switch_"}   # genome names that are C keywords get a trailing underscore


def _cname(n: str) -> str:
    return C_RESERVED.get(n, n)


def generate_header() -> str:
    """C struct definitions mirroring the ctypes structures above (written to csim/us_structs.h)."""
    def body(fields):
        out = []
        for n, k in fields:
            n = _cname(n)
            out.append(f"    double {n}[{k}];" if k > 1 else f"    double {n};")
        return "\n".join(out)
    res_ints = "\n".join(f"    int32_t {n};" for n in RESULT_INT_FIELDS)
    if len(RESULT_INT_FIELDS) % 2:
        res_ints += "\n    int32_t _pad;"
    res_dbls = "\n".join(f"    double {n};" for n in RESULT_DBL_FIELDS)
    outc = "\n".join(f"#define US_{o.upper()} {i}" for i, o in enumerate(OUTCOMES))
    return f"""/* GENERATED by scripts/build_c.py from ultisim/cabi.py - do not edit by hand. */
#ifndef US_STRUCTS_H
#define US_STRUCTS_H
#include <stdint.h>

#define ULTISIM_ABI_VERSION {ABI_VERSION}
#define US_N_MODEL_W {N_MODEL_W}
#define US_N_OUTCOMES {len(OUTCOMES)}
{outc}

typedef struct {{
{body([(n, 1) for n in PLAYER_FIELDS])}
}} UsPlayer;

typedef struct {{
{body([(n, 1) for n, _ in CONFIG_FIELDS])}
}} UsConfig;

typedef struct {{
{body(OFFENSE_FIELDS)}
}} UsOffense;

typedef struct {{
{body(DEFENSE_FIELDS)}
}} UsDefense;

typedef struct {{
    UsPlayer players[14];
    int32_t off_idx[2];   /* offense genome index used by team 0 / team 1 */
    int32_t def_idx[2];   /* defense genome index used by team 0 / team 1 */
    int32_t mode;         /* 0 = one possession (team 0 attacks), 1 = full point */
    int32_t _pad;
    uint64_t seed;
    double start_x, start_y;  /* start_y < 0: random */
}} UsGameSpec;

typedef struct {{
{res_ints}
{res_dbls}
}} UsResult;

#endif
"""


@lru_cache(maxsize=1)
def lib() -> ctypes.CDLL:
    name = "ultisim_core.dll" if os.name == "nt" else "ultisim_core.so"
    path = BUILD / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/build_c.py")
    if os.name == "nt":
        global _DLL_DIR_HANDLE  # keep alive: closing it removes the search path
        _DLL_DIR_HANDLE = os.add_dll_directory(str(BUILD))  # MinGW runtime DLLs live here
    so = ctypes.CDLL(str(path))
    so.us_abi_version.restype = ctypes.c_int32
    so.us_max_threads.restype = ctypes.c_int32
    so.us_has_openmp.restype = ctypes.c_int32
    so.us_smoke.restype = ctypes.c_double
    so.us_smoke.argtypes = [ctypes.c_int64]
    got = so.us_abi_version()
    if got != ABI_VERSION:
        raise RuntimeError(f"C core ABI {got} != Python ABI {ABI_VERSION}; rebuild (python scripts/build_c.py)")
    for fn, size in (("us_sizeof_config", ctypes.sizeof(UsConfig)), ("us_sizeof_spec", ctypes.sizeof(UsGameSpec)),
                     ("us_sizeof_result", ctypes.sizeof(UsResult)), ("us_sizeof_offense", ctypes.sizeof(UsOffense)),
                     ("us_sizeof_defense", ctypes.sizeof(UsDefense))):
        getattr(so, fn).restype = ctypes.c_int64
        if getattr(so, fn)() != size:
            raise RuntimeError(f"{fn}: C {getattr(so, fn)()} != Python {size}; rebuild")
    P = ctypes.POINTER
    D_ = ctypes.c_double
    so.us_sim_batch.restype = ctypes.c_int32
    so.us_sim_batch.argtypes = [P(UsConfig), P(UsOffense), ctypes.c_int32, P(UsDefense), ctypes.c_int32,
                                P(UsGameSpec), ctypes.c_int32, P(UsResult), ctypes.c_int32]
    so.us_time_to_reach.restype = D_
    so.us_time_to_reach.argtypes = [D_] * 12
    so.us_flight_point.restype = None
    so.us_flight_point.argtypes = [D_] * 9 + [P(D_)]
    so.us_error_sigma.restype = D_
    so.us_error_sigma.argtypes = [P(UsConfig), D_, ctypes.c_int32, D_, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    so.us_mark_block_prob.restype = D_
    so.us_mark_block_prob.argtypes = [P(UsConfig), D_, D_, D_, D_, ctypes.c_int32, D_, D_, D_, D_, ctypes.c_int32]
    so.us_step.restype = None
    so.us_step.argtypes = [ctypes.c_int32] + [P(D_)] * 11 + [D_, P(ctypes.c_int32)]
    so.us_flight_height.restype = D_
    so.us_flight_height.argtypes = [P(UsConfig), D_, D_, D_, D_]
    so.us_hungarian.restype = None
    so.us_hungarian.argtypes = [ctypes.c_int32, ctypes.c_int32, P(D_), P(ctypes.c_int32)]
    return so
