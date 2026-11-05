/* ultisim C engine: a function-by-function port of ultisim/engine_ref.py.
 *
 * Same rules, same order of operations, same formulas. Differences from the reference:
 *   - its own RNG (xoshiro256**), so individual games differ; statistics must match (tests/test_parity.py)
 *   - no replay recording or throw log (champion replays are rendered with the Python reference)
 * Frame: the engine works in the current offense's frame (offense attacks +x); see engine_ref.py.
 */
#include <math.h>
#include <string.h>
#include <float.h>
#include "ultisim.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)

/* player modes (codes 0-8 shared with the viewer; JAB/RESET internal) */
enum { SET = 0, CUT = 1, CLEAR = 2, CHASE = 3, GROUND = 4, MARK = 5, HOLD = 6, DEF = 7, ZONE = 8, JAB = 101, RESET = 102 };
enum { BH = 0, FH = 1, HAM = 2 };
enum { F_FOREHAND = 0, F_BACKHAND = 1, F_STRAIGHT = 2, F_MIDDLE = 3, F_SIDELINE = 4 };

#define NP 14
#define NSLOT 6
#define KS 10           /* samples along a candidate flight path */
#define KCH 18          /* samples along a live flight for chasers */
#define NMENU 9
#define MAXC (12 * NMENU) /* 6 receivers x 2 leads x 9 throws */
#define HCAP 16

/* ------------------------------------------------------------------ small math */
static inline double sq(double x) { return x * x; }
static inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline double maxd(double a, double b) { return a > b ? a : b; }
static inline double mind(double a, double b) { return a < b ? a : b; }
static inline double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double ndtr_(double x) { return 0.5 * erfc(-x * 0.70710678118654752440); }
static inline double wrap_deg(double a) {
    double r = fmod(a + 180.0, 360.0);
    if (r < 0) r += 360.0;
    return r - 180.0;
}

/* ------------------------------------------------------------------ RNG: xoshiro256** */
typedef struct { uint64_t s[4]; int has_spare; double spare; } Rng;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static void rng_seed(Rng *r, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
    r->has_spare = 0;
}
static inline uint64_t rng_next(Rng *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static inline double rng_u(Rng *r) { return (double)(rng_next(r) >> 11) * 0x1.0p-53; }
static double rng_n(Rng *r) {
    if (r->has_spare) { r->has_spare = 0; return r->spare; }
    double u, v, s;
    do { u = 2.0 * rng_u(r) - 1.0; v = 2.0 * rng_u(r) - 1.0; s = u * u + v * v; } while (s >= 1.0 || s == 0.0);
    double m = sqrt(-2.0 * log(s) / s);
    r->spare = v * m; r->has_spare = 1;
    return u * m;
}

/* ------------------------------------------------------------------ Hungarian (n <= m <= 7) */
void us_hungarian(int32_t n, int32_t m, const double *a, int32_t *row_to_col) {
    double u[9] = {0}, v[9] = {0}, minv[9];
    int p[9] = {0}, way[9] = {0}, used[9];
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        for (int j = 0; j <= m; j++) { minv[j] = DBL_MAX; used[j] = 0; }
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            double delta = DBL_MAX;
            for (int j = 1; j <= m; j++) {
                if (used[j]) continue;
                double cur = a[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    for (int j = 1; j <= m; j++) if (p[j]) row_to_col[p[j] - 1] = j - 1;
}

/* ------------------------------------------------------------------ physics building blocks */
double us_time_to_reach(double px, double py, double vx, double vy, double top, double accel,
                        double brake, double reach, double qx, double qy, double delay, double extra) {
    double dx = qx - px, dy = qy - py;
    double d = sqrt(dx * dx + dy * dy);
    double d_eff = d - reach - extra;
    if (d_eff <= 0.0) return 0.0;
    double inv = 1.0 / (d > 1e-9 ? d : 1e-9);
    double ux = dx * inv, uy = dy * inv;
    double v_par = vx * ux + vy * uy;
    double v_perp = fabs(vx * uy - vy * ux);
    double t_turn = (maxd(-v_par, 0.0) + 0.5 * v_perp) / brake;
    double v0 = maxd(v_par, 0.0);
    double d_acc = (top * top - v0 * v0) / (2.0 * accel);
    double t_run;
    if (d_eff <= d_acc) t_run = (-v0 + sqrt(v0 * v0 + 2.0 * accel * d_eff)) / accel;
    else t_run = (top - v0) / accel + (d_eff - d_acc) / top;
    return t_turn + t_run + delay;
}

static inline void bezier(double p0x, double p0y, double cx, double cy, double p1x, double p1y, double s,
                          double *ox, double *oy) {
    double a = (1 - s) * (1 - s), b = 2 * (1 - s) * s, c = s * s;
    *ox = a * p0x + b * cx + c * p1x;
    *oy = a * p0y + b * cy + c * p1y;
}

void us_flight_point(double p0x, double p0y, double cx, double cy, double p1x, double p1y,
                     double T, double s, double tail, double *out2) {
    double bx, by;
    bezier(p0x, p0y, cx, cy, p1x, p1y, mind(s, 1.0), &bx, &by);
    double tx = p1x - cx, ty = p1y - cy, tn = sqrt(tx * tx + ty * ty);
    double ux = tn > 1e-9 ? tx / tn : 0.0, uy = tn > 1e-9 ? ty / tn : 0.0;
    double v_end = 2.0 * tn / maxd(T, 1e-6);
    double e = clampd((s - 1.0) * T, 0.0, tail);
    double glide = v_end * (e - 0.35 * e * e / maxd(tail, 1e-6));
    out2[0] = bx + ux * glide;
    out2[1] = by + uy * glide;
}

static void control_point(const UsConfig *c, double p0x, double p0y, double p1x, double p1y, double curve,
                          double *cx, double *cy) {
    double dx = p1x - p0x, dy = p1y - p0y, d = sqrt(dx * dx + dy * dy);
    double inv = 1.0 / maxd(d, 1e-9);
    double lx = -dy * inv, ly = dx * inv;
    double k = curve * c->curve_amount * d;
    *cx = 0.5 * (p0x + p1x) + lx * k;
    *cy = 0.5 * (p0y + p1y) + ly * k;
}

static inline double type_err_mult(const UsConfig *c, int t) { return t == BH ? c->bh_err_mult : (t == FH ? c->fh_err_mult : c->ham_err_mult); }
static inline double type_mark_mult(const UsConfig *c, int t) { return t == BH ? c->bh_mark_mult : (t == FH ? c->fh_mark_mult : c->ham_mark_mult); }
static inline double type_side(const UsConfig *c, int t) { return t == BH ? c->bh_release_side : (t == FH ? c->fh_release_side : c->ham_release_side); }

static double flight_time(const UsConfig *c, int t, double d, int huck, int loft) {
    if (t == HAM) return c->ham_hang + d / c->ham_speed;
    double sp = t == BH ? c->bh_speed : c->fh_speed, hg = t == BH ? c->bh_hang : c->fh_hang;
    if (huck) return t == BH ? c->bh_huck_hang + d / c->bh_huck_speed : c->fh_huck_hang + d / c->fh_huck_speed;
    if (loft) return hg + c->loft_hang_extra + d / (sp * c->loft_speed_mult);   /* touch pass: slower, hangs */
    return hg + d / sp;
}

/* peak height (m): flat bullets stay low, lofted touch passes rise to 3-4 m, hucks and hammers go higher */
static double throw_peak(const UsConfig *c, int t, double d, int huck, int loft) {
    if (t == HAM) return c->hammer_hbase + c->hammer_hper_m * d;
    if (huck) return c->huck_hbase + c->huck_hper_m * d;
    if (loft) return mind(c->loft_hbase + c->loft_hper_m * d, c->loft_hmax);
    return c->flat_hbase + c->flat_hper_m * d;
}

/* disc height at flight fraction s: quadratic release -> peak (s = 0.5) -> catch; the tail sinks */
static double flight_height(const UsConfig *c, double peak, double s, double T, double h0) {
    double h1 = c->h_catch, ctrl = 2.0 * peak - 0.5 * (h0 + h1);
    if (s > 1.0) return h1 - (h1 - c->h_ground) * clampd((s - 1.0) * T / c->tail_time, 0.0, 1.0);
    double sc = clampd(s, 0.0, 1.0);
    return (1 - sc) * (1 - sc) * h0 + 2 * (1 - sc) * sc * ctrl + sc * sc * h1;
}

double us_error_sigma(const UsConfig *c, double skill, int32_t t, double d, int32_t is_break, int32_t marked, int32_t curved) {
    double skill_mult = c->skill_err_worst + (c->skill_err_best - c->skill_err_worst) * skill;
    double sig = (c->err_a + c->err_b * d + c->err_c * d * d) * skill_mult * type_err_mult(c, t);
    if (is_break) sig *= c->break_err_mult;
    if (marked) sig *= c->marked_err_mult;
    if (curved) sig *= c->curve_err_mult;
    return sig;
}

double us_mark_block_prob(const UsConfig *c, double hx, double hy, double ex, double ey, int32_t t,
                          double center, double width, double marking, double break_skill, int32_t fake_active) {
    double ang = atan2(ey - hy, ex - hx) * RAD2DEG;
    double delta = fabs(wrap_deg(ang - center));
    double cover = clampd(1.0 - sq(delta / maxd(width, 1e-6)), 0.0, 1.0);
    double skill = (1.0 + c->mark_skill_weight * (2.0 * marking - 1.0)) *
                   (1.0 - c->break_skill_weight * (2.0 * break_skill - 1.0) * 0.5);
    double fake = fake_active ? (0.35 + 0.4 * marking) : 1.0;
    double spread = mind(1.0, c->mark_width_ref / maxd(width, 1e-6));  /* fixed blocking capacity */
    /* break-side tail: the marker reaches for anything released toward their side, fading out at mark_tail_deg */
    double tail = c->mark_break_tail * clampd(1.0 - delta / c->mark_tail_deg, 0.0, 1.0) * (1.0 - cover);
    return clampd(c->mark_block_max * (spread * cover + tail) * skill * type_mark_mult(c, t) * fake, 0.0, 0.95);
}

static inline void step_one(double *px, double *py, double *vx, double *vy, double tx, double ty, double speed,
                            double accel, double brake, double top, double tvx, double tvy, double dt, int frozen) {
    if (frozen) { *vx = 0.0; *vy = 0.0; return; }
    double dx = tx - *px, dy = ty - *py, dist = sqrt(dx * dx + dy * dy);
    double safe = maxd(dist, 1e-9);
    double cap = mind(speed, top);
    double want = mind(cap, sqrt(2.0 * brake * maxd(dist - 0.05, 0.0)));
    double vdx = dx / safe * want + tvx, vdy = dy / safe * want + tvy;
    double vd = sqrt(vdx * vdx + vdy * vdy);
    if (vd > cap) { double k = cap / vd; vdx *= k; vdy *= k; }
    double dvx = vdx - *vx, dvy = vdy - *vy;
    double vmag = sqrt(*vx * *vx + *vy * *vy);
    double hx = vmag > 1e-6 ? *vx / vmag : 0.0, hy = vmag > 1e-6 ? *vy / vmag : 0.0;
    double along = dvx * hx + dvy * hy;
    double amax = along < 0.0 ? brake : accel;
    double dvmag = sqrt(dvx * dvx + dvy * dvy);
    double scale = mind(1.0, amax * dt / maxd(dvmag, 1e-9));
    double nvx = *vx + dvx * scale, nvy = *vy + dvy * scale;
    double nv = sqrt(nvx * nvx + nvy * nvy);
    if (nv > top) { nvx *= top / nv; nvy *= top / nv; }
    *px += 0.5 * (*vx + nvx) * dt;
    *py += 0.5 * (*vy + nvy) * dt;
    *vx = nvx; *vy = nvy;
}

void us_step(int32_t n, double *px, double *py, double *vx, double *vy, const double *tx, const double *ty,
             const double *speed, const double *accel, const double *brake, const double *top,
             const double *tvel_xy, double dt, const int32_t *frozen) {
    for (int i = 0; i < n; i++)
        step_one(&px[i], &py[i], &vx[i], &vy[i], tx[i], ty[i], speed[i], accel[i], brake[i], top[i],
                 tvel_xy ? tvel_xy[2 * i] : 0.0, tvel_xy ? tvel_xy[2 * i + 1] : 0.0, dt, frozen ? frozen[i] : 0);
}

double us_flight_height(const UsConfig *c, double peak, double s, double T, double h0) { return flight_height(c, peak, s, T, h0); }

/* ------------------------------------------------------------------ game state */
typedef struct {
    int thrower, receiver, ttype, huck, curve, is_break;
    double p0x, p0y, cx, cy, p1x, p1y, ix, iy;   /* release, control, actual landing, intended */
    double t0, T, lo, hi, dist, s_end, prevx, prevy, peak, h0;
    int loft;
    int bid_done, touched;                        /* bitmasks over players */
} Flight;

typedef struct {
    int outcome, offense, throws, completions, hucks, hcomp, breaks, bcomp, hammers, fakes;
    double duration, max_stall, end_x, end_y, sum_stall, sum_p;
} Poss;

typedef struct {
    const UsConfig *c;
    const UsPlayer *P;
    const UsOffense *og[2];
    const UsDefense *dg[2];
    int force[2], nzone[2];
    Rng rng;
    double brake[NP], reach[NP], effh[NP], rng_bh[NP], rng_fh[NP], rng_ham[NP], standing[NP], maxreach[NP];
    double px[NP], py[NP], vx[NP], vy[NP];
    int mode[NP];
    double mtx[NP], mty[NP], cex[NP], cey[NP], muntil[NP], ground_until[NP], next_cut[NP];
    int slot[NP], zslot[NP], match[NP], def_of[NP];
    double t;
    int offense, holder, in_flight, reset_player;
    double dx, dy;
    Flight f;
    double stall, ready_time, grace_until, next_decision, fake_until, fake_effect_until, last_fake, s_open, poss_t0;
    double next_switch;
    int keff;               /* zone players right now (zone-to-person transition can drop it to 0) */
    int hmax, hlen, hhead;
    double hpx[HCAP][NP], hpy[HCAP][NP], hvx[HCAP][NP], hvy[HCAP][NP];
    Poss res;
    double model_w[US_N_MODEL_W];
} Game;

#define TOP(i) (G->P[i].top_speed)
#define O0 (7 * G->offense)
#define D0 (7 * (1 - G->offense))
#define OG (G->og[G->offense])
#define DG (G->dg[1 - G->offense])
#define FORCE (G->force[1 - G->offense])
#define KZONE (G->nzone[1 - G->offense])   /* 0 person, 7 zone, 1..6 flex */

static inline void clip_field(const Game *G, double *x, double *y, double margin) {
    *x = clampd(*x, margin, G->c->L - margin);
    *y = clampd(*y, margin, G->c->W - margin);
}
static inline double dist2(double ax, double ay, double bx, double by) { return sqrt(sq(ax - bx) + sq(ay - by)); }

static double compute_s_open(const Game *G, double y) {
    double toward_mid = y <= G->c->W / 2 ? 1.0 : -1.0;
    switch (FORCE) {
        case F_FOREHAND: return -1.0;
        case F_BACKHAND: return 1.0;
        case F_SIDELINE: return -toward_mid;
        default: return toward_mid;
    }
}

static inline void disc_ref(const Game *G, double *x, double *y) {
    if (G->in_flight) { *x = G->f.ix; *y = G->f.iy; } else { *x = G->dx; *y = G->dy; }
}

static void mark_geometry(const Game *G, double *center, double *width) {
    double ma = FORCE == F_STRAIGHT ? 0.0 : DG->mark_angle;
    double c = atan2(-G->s_open * sin(ma * DEG2RAD), cos(ma * DEG2RAD)) * RAD2DEG;
    if (G->t < G->fake_effect_until) c += G->s_open * G->c->fake_shift_deg;
    *center = c;
    *width = atan2(G->c->mark_reach, DG->mark_dist) * RAD2DEG;  /* body-based coverage */
}

/* possession urgency: after urgency_start s, patience and risk aversion fade to urgency_floor */
static inline double urgency(const Game *G) {
    const UsConfig *c = G->c;
    double u = clampd((G->t - G->poss_t0 - c->urgency_start) / c->urgency_span, 0.0, 1.0);
    return 1.0 - (1.0 - c->urgency_floor) * u;
}

static inline double mark_quality(const Game *G) { return 1.0 - exp(-DG->mark_dist / G->c->mark_react_dist); }

static int marker_id(const Game *G) {
    if (G->holder < 0) return -1;
    if (G->keff >= 1) {   /* zone / flex: whoever holds zone spot 0 marks */
        for (int i = D0; i < D0 + 7; i++) if (G->zslot[i] == 0) return i;
        return -1;
    }
    return G->def_of[G->holder];
}

static int marker_set(const Game *G) {
    int mk = marker_id(G);
    return mk >= 0 && dist2(G->px[mk], G->py[mk], G->px[G->holder], G->py[G->holder]) <= G->c->mark_set_radius;
}

static void hist_push(Game *G) {
    int k = G->hhead;
    memcpy(G->hpx[k], G->px, sizeof G->px); memcpy(G->hpy[k], G->py, sizeof G->py);
    memcpy(G->hvx[k], G->vx, sizeof G->vx); memcpy(G->hvy[k], G->vy, sizeof G->vy);
    G->hhead = (k + 1) % G->hmax;
    if (G->hlen < G->hmax) G->hlen++;
}
static inline int hist_idx(const Game *G, int back) { return ((G->hhead - 1 - back) % G->hmax + G->hmax) % G->hmax; }

static void delayed(const Game *G, int id, double delay, double *ox, double *oy, double *ovx, double *ovy) {
    if (G->hlen == 0) { *ox = G->px[id]; *oy = G->py[id]; *ovx = G->vx[id]; *ovy = G->vy[id]; return; }
    int H = G->hlen;
    double k = clampd(delay / G->c->dt, 0.0, maxd(H - 1.000001, 0.0));
    int k0 = (int)floor(k);
    if (k0 > H - 1) k0 = H - 1;
    if (k0 < 0) k0 = 0;
    double w = k - k0;
    int a = hist_idx(G, k0), b = hist_idx(G, k0 + 1 < H - 1 ? k0 + 1 : H - 1);
    *ox = (1 - w) * G->hpx[a][id] + w * G->hpx[b][id];
    *oy = (1 - w) * G->hpy[a][id] + w * G->hpy[b][id];
    *ovx = (1 - w) * G->hvx[a][id] + w * G->hvx[b][id];
    *ovy = (1 - w) * G->hvy[a][id] + w * G->hvy[b][id];
}

/* ------------------------------------------------------------------ formation / assignment */
static void homes(const Game *G, double rx, double ry, double *hx, double *hy) {
    const UsOffense *g = OG;
    for (int k = 0; k < NSLOT; k++) {
        double x = rx + g->home_dx[k];
        double yd = ry + G->s_open * g->home_dy[k];
        double ym = G->c->W / 2 + G->s_open * g->home_dy[k];
        double y = (1 - g->y_anchor[k]) * yd + g->y_anchor[k] * ym;
        clip_field(G, &x, &y, 1.2);
        hx[k] = x; hy[k] = y;
    }
}

static void assign_slots(Game *G, const int *players, double rx, double ry) {
    double hx[NSLOT], hy[NSLOT], cost[NSLOT * NSLOT];
    int r2c[NSLOT];
    homes(G, rx, ry, hx, hy);
    const UsOffense *g = OG;
    for (int i = 0; i < NSLOT; i++) {
        int p = players[i];
        for (int k = 0; k < NSLOT; k++)
            cost[i * NSLOT + k] = dist2(G->px[p], G->py[p], hx[k], hy[k]) +
                                  6.0 * (g->handler[k] > 0.5) * (1.0 - G->P[p].throw_skill);
    }
    us_hungarian(NSLOT, NSLOT, cost, r2c);
    for (int i = O0; i < O0 + 7; i++) G->slot[i] = -1;
    for (int i = 0; i < NSLOT; i++) G->slot[players[i]] = r2c[i];
}

static void zone_points(const Game *G, double rx, double ry, double *zx, double *zy) {
    const UsDefense *g = DG;
    double mc, mw;
    mark_geometry(G, &mc, &mw);
    zx[0] = rx + g->mark_dist * cos(mc * DEG2RAD);
    zy[0] = ry + g->mark_dist * sin(mc * DEG2RAD);
    for (int k = 0; k < NSLOT; k++) { zx[k + 1] = rx + g->zone_dx[k]; zy[k + 1] = ry + G->s_open * g->zone_dy[k]; }
    if (g->trap > 0.0) {   /* sideline trap: short zone spots collapse toward the thrower */
        double edge = mind(ry, G->c->W - ry);
        double tf = g->trap * clampd(1.0 - edge / G->c->trap_zone, 0.0, 1.0);
        if (tf > 0.0)
            for (int k = 1; k < 7; k++)
                if (dist2(zx[k], zy[k], rx, ry) < 15.0) { zx[k] += 0.4 * tf * (rx - zx[k]); zy[k] += 0.4 * tf * (ry - zy[k]); }
    }
    for (int k = 0; k < 7; k++) clip_field(G, &zx[k], &zy[k], 0.8);
}

static void assign_zone(Game *G) {
    double rx, ry, zx[7], zy[7], cost[49];
    int r2c[7];
    disc_ref(G, &rx, &ry);
    zone_points(G, rx, ry, zx, zy);
    for (int i = 0; i < 7; i++)
        for (int k = 0; k < 7; k++) cost[i * 7 + k] = dist2(G->px[D0 + i], G->py[D0 + i], zx[k], zy[k]);
    us_hungarian(7, 7, cost, r2c);
    for (int i = 0; i < 7; i++) G->zslot[D0 + i] = r2c[i];
}

static void assign_matchups(Game *G, int by_distance) {
    double cost[49];
    int r2c[7];
    const UsDefense *g = DG;
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) {
            int d = D0 + i, o = O0 + j;
            double c = g->match_speed * fabs(TOP(d) - TOP(o));
            if (by_distance) c += 0.3 * dist2(G->px[d], G->py[d], G->px[o], G->py[o]);
            cost[i * 7 + j] = c;
        }
    us_hungarian(7, 7, cost, r2c);
    for (int i = 0; i < NP; i++) { G->match[i] = -1; G->def_of[i] = -1; }
    for (int i = 0; i < 7; i++) { G->match[D0 + i] = O0 + r2c[i]; G->def_of[O0 + r2c[i]] = D0 + i; }
}

static void clear_matchups(Game *G) {
    for (int i = 0; i < NP; i++) { G->match[i] = -1; G->def_of[i] = -1; }
}

static inline int rest_mode(const Game *G, int d) { return G->zslot[d] >= 0 ? ZONE : DEF; }

static inline double help_delay(const Game *G, int d, int receiver) {
    return (G->match[d] >= 0 && G->match[d] != receiver) ? G->c->help_read_delay : 0.0;
}

/* flex: k defenders take zone spots 0..k-1 (spot 0 = marker); 7-k play person on the cutters chosen by
 * flex_focus (deepest first at 1, closest to the disc first at 0). */
static void assign_flex(Game *G, int k) {
    const UsDefense *g = DG;
    double rx, ry, zx[7], zy[7], cx[7], cy[7], score[6], cost[49];
    int att[6], na = 0, r2c[7], chosen[7];
    disc_ref(G, &rx, &ry);
    zone_points(G, rx, ry, zx, zy);
    for (int a = O0; a < O0 + 7; a++) if (a != G->holder) { att[na] = a; score[na] = (2.0 * g->flex_focus - 1.0) * (G->px[a] - rx); na++; }
    int order[6];
    for (int i = 0; i < na; i++) order[i] = i;
    for (int i = 1; i < na; i++) {   /* stable insertion sort, descending score */
        int v = order[i], j = i - 1;
        while (j >= 0 && score[order[j]] < score[v]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
    }
    for (int j = 0; j < k; j++) { cx[j] = zx[j]; cy[j] = zy[j]; }
    for (int j = 0; j < 7 - k; j++) { chosen[j] = att[order[j]]; cx[k + j] = G->px[chosen[j]]; cy[k + j] = G->py[chosen[j]]; }
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) cost[i * 7 + j] = dist2(G->px[D0 + i], G->py[D0 + i], cx[j], cy[j]);
    us_hungarian(7, 7, cost, r2c);
    for (int i = 0; i < 7; i++) {
        int d = D0 + i, j = r2c[i];
        if (j < k) G->zslot[d] = j;
        else { G->match[d] = chosen[j - k]; G->def_of[chosen[j - k]] = d; }
    }
}

static void assign_defense(Game *G, int by_distance) {
    int k = G->keff;
    for (int i = D0; i < D0 + 7; i++) G->zslot[i] = -1;
    clear_matchups(G);
    if (k == 0) assign_matchups(G, by_distance);
    else if (k == 7) assign_zone(G);
    else assign_flex(G, k);
    G->next_switch = G->t + G->c->switch_interval;
}

static void defense_targets(Game *G, double *tx, double *ty, double *spd, double *tvx, double *tvy);

static void start_possession(Game *G, double dx, double dy, int fresh) {
    const UsConfig *c = G->c;
    G->in_flight = 0;
    G->dx = dx; G->dy = dy;
    G->s_open = compute_s_open(G, dy);
    G->stall = 0.0;
    G->reset_player = -1;
    G->fake_until = G->fake_effect_until = -1.0;
    int others[NSLOT];
    double grace;
    if (fresh) {
        const UsOffense *g = OG;
        double hx[NSLOT], hy[NSLOT], sx[7], sy[7], hd[7], cost[49];
        int r2c[7];
        homes(G, dx, dy, hx, hy);
        sx[0] = dx; sy[0] = dy; hd[0] = 1.0;
        for (int k = 0; k < NSLOT; k++) { sx[k + 1] = hx[k]; sy[k + 1] = hy[k]; hd[k + 1] = g->handler[k] > 0.5; }
        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 7; j++) cost[i * 7 + j] = -G->P[O0 + i].throw_skill * hd[j] * 3.0 + rng_u(&G->rng) * 0.1;
        us_hungarian(7, 7, cost, r2c);
        for (int i = 0; i < 7; i++) {
            int p = O0 + i;
            G->px[p] = sx[r2c[i]]; G->py[p] = sy[r2c[i]]; G->vx[p] = G->vy[p] = 0.0;
            if (r2c[i] == 0) G->holder = p;
        }
        grace = 0.0;
    } else {
        int best = O0;
        double bd = DBL_MAX;
        for (int i = O0; i < O0 + 7; i++) {
            double d = dist2(G->px[i], G->py[i], dx, dy);
            if (d < bd) { bd = d; best = i; }
        }
        G->holder = best;
        G->px[best] = dx; G->py[best] = dy; G->vx[best] = G->vy[best] = 0.0;
        grace = c->pickup_grace;
    }
    int n = 0;
    for (int i = O0; i < O0 + 7; i++) if (i != G->holder) others[n++] = i;
    assign_slots(G, others, dx, dy);
    for (int i = O0; i < O0 + 7; i++) { G->mode[i] = SET; G->next_cut[i] = 0.0; }
    G->mode[G->holder] = HOLD;
    for (int i = 0; i < NP; i++) G->ground_until[i] = -1.0;
    G->ready_time = G->t + G->P[G->holder].release_time + grace;
    G->grace_until = G->t + grace;
    G->next_decision = G->ready_time;
    G->keff = KZONE;
    int k = G->keff;
    if (fresh && k >= 1) {   /* zone spots first (as if set after the pull), then assign by distance */
        double zx[7], zy[7];
        int perm[7] = {0, 1, 2, 3, 4, 5, 6};
        zone_points(G, dx, dy, zx, zy);
        for (int i = 6; i > 0; i--) { int j = (int)(rng_u(&G->rng) * (i + 1)); int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }
        for (int i = 0; i < 7; i++) { int d = D0 + i; G->px[d] = zx[perm[i]]; G->py[d] = zy[perm[i]]; G->vx[d] = G->vy[d] = 0.0; }
    }
    assign_defense(G, !(fresh && k == 0));
    if (fresh && k < 7) {    /* person-role defenders start in their scheme positions */
        for (int i = D0; i < D0 + 7; i++) {
            int a = G->match[i];
            if (a >= 0) { G->px[i] = G->px[a] + 1.5; G->py[i] = G->py[a]; }
            G->vx[i] = G->vy[i] = 0.0;
        }
        G->hlen = 0;
        for (int rep = 0; rep < 2; rep++) {
            double tx[NP], ty[NP], sp[NP];
            for (int i = 0; i < NP; i++) { tx[i] = G->px[i]; ty[i] = G->py[i]; sp[i] = 0.0; }
            for (int i = D0; i < D0 + 7; i++) G->mode[i] = rest_mode(G, i);
            defense_targets(G, tx, ty, sp, NULL, NULL);
            for (int i = D0; i < D0 + 7; i++) { G->px[i] = tx[i]; G->py[i] = ty[i]; }
        }
    }
    for (int i = D0; i < D0 + 7; i++) G->mode[i] = rest_mode(G, i);
    G->hlen = 0;
    G->poss_t0 = G->t;
    memset(&G->res, 0, sizeof G->res);
    G->res.outcome = -1;
    G->res.offense = G->offense;
}

/* ------------------------------------------------------------------ offense behaviour */
static void begin_cut(Game *G, int i, int k) {
    const UsOffense *g = OG;
    double ax = cos(g->cutA_ang[k] * DEG2RAD), ay = G->s_open * sin(g->cutA_ang[k] * DEG2RAD);
    double bx = cos(g->cutB_ang[k] * DEG2RAD), by = G->s_open * sin(g->cutB_ang[k] * DEG2RAD);
    double awx = 0.0, awy = 0.0, vx_, vy_;
    int d = G->def_of[i];
    if (d >= 0) { vx_ = G->px[i] - G->px[d]; vy_ = G->py[i] - G->py[d]; }
    else {
        int dn = D0; double bd = DBL_MAX;
        for (int j = D0; j < D0 + 7; j++) { double dd = dist2(G->px[j], G->py[j], G->px[i], G->py[i]); if (dd < bd) { bd = dd; dn = j; } }
        vx_ = G->px[i] - G->px[dn]; vy_ = G->py[i] - G->py[dn];
    }
    double nv = sqrt(vx_ * vx_ + vy_ * vy_);
    if (nv > 0.3) { awx = vx_ / nv; awy = vy_ / nv; }
    double la = g->cutA_bias[k] + g->read_def[k] * (ax * awx + ay * awy);
    double lb = g->read_def[k] * (bx * awx + by * awy);
    double dxu, dyu, ln;
    if (rng_u(&G->rng) < sigm(la - lb)) { dxu = ax; dyu = ay; ln = g->cutA_len[k]; }
    else { dxu = bx; dyu = by; ln = g->cutB_len[k]; }
    if (g->seek_space[k] > 0.0) {   /* bend the cut toward open space, away from nearby defenders */
        double svx = 0.0, svy = 0.0;
        int any = 0;
        for (int j = D0; j < D0 + 7; j++) {
            double rx_ = G->px[i] - G->px[j], ry_ = G->py[i] - G->py[j], dd = sqrt(rx_ * rx_ + ry_ * ry_);
            if (dd < 10.0) { double w = 1.0 / sq(maxd(dd, 0.5)); svx += rx_ * w; svy += ry_ * w; any = 1; }
        }
        double sn = sqrt(svx * svx + svy * svy);
        if (any && sn > 1e-6) {
            double w = g->seek_space[k];
            dxu = (1.0 - w) * dxu + w * svx / sn;
            dyu = (1.0 - w) * dyu + w * svy / sn;
            double dn = maxd(sqrt(dxu * dxu + dyu * dyu), 1e-6);
            dxu /= dn; dyu /= dn;
        }
    }
    double ex = G->px[i] + dxu * ln, ey = G->py[i] + dyu * ln;
    clip_field(G, &ex, &ey, 0.7);
    G->cex[i] = ex; G->cey[i] = ey;
    if (rng_u(&G->rng) < g->jab[k]) {
        double jx = G->px[i] - dxu * 1.5, jy = G->py[i] - dyu * 1.5;
        clip_field(G, &jx, &jy, 0.8);
        G->mode[i] = JAB; G->mtx[i] = jx; G->mty[i] = jy; G->muntil[i] = G->t + 0.35;
    } else {
        G->mode[i] = CUT; G->mtx[i] = ex; G->mty[i] = ey; G->muntil[i] = G->t + ln / TOP(i) + 1.2;
    }
}

static void offense_targets(Game *G, double *tx, double *ty, double *spd) {
    const UsOffense *g = OG;
    const double jog = G->c->jog_fraction;
    double rx, ry, hx[NSLOT], hy[NSLOT];
    disc_ref(G, &rx, &ry);
    homes(G, rx, ry, hx, hy);
    int have_disc = G->holder >= 0;
    if (have_disc && G->stall >= g->reset_stall && G->reset_player < 0) {
        double rsx = G->dx + g->reset_dx, rsy = G->dy + G->s_open * g->reset_dy, bd = DBL_MAX;
        int best = -1;
        for (int i = O0; i < O0 + 7; i++) {
            if (i == G->holder || G->slot[i] < 0 || !(g->handler[G->slot[i]] > 0.5) || G->mode[i] == GROUND) continue;
            double d = dist2(G->px[i], G->py[i], rsx, rsy);
            if (d < bd) { bd = d; best = i; }
        }
        if (best >= 0) { G->reset_player = best; G->mode[best] = RESET; }
    }
    for (int i = O0; i < O0 + 7; i++) {
        int md = G->mode[i];
        if (G->t < G->ground_until[i]) { tx[i] = G->px[i]; ty[i] = G->py[i]; spd[i] = 0.0; continue; }
        if (md == GROUND) G->mode[i] = md = CLEAR;
        if (md == HOLD) { tx[i] = G->px[i]; ty[i] = G->py[i]; spd[i] = 0.0; continue; }
        if (md == CHASE) { tx[i] = G->mtx[i]; ty[i] = G->mty[i]; spd[i] = TOP(i); continue; }
        int k = G->slot[i];
        if (k < 0) { tx[i] = G->px[i]; ty[i] = G->py[i]; spd[i] = 0.0; continue; }
        double homex = hx[k], homey = hy[k];
        if (md == RESET) {
            double rsx = G->dx + g->reset_dx, rsy = G->dy + G->s_open * g->reset_dy;
            clip_field(G, &rsx, &rsy, 0.8);
            tx[i] = rsx; ty[i] = rsy; spd[i] = TOP(i);
            continue;
        }
        if (md == SET) {
            if (have_disc && G->t >= G->ready_time - 0.3 + g->cut_delay[k] && G->t >= G->next_cut[i]) {
                begin_cut(G, i, k);
                md = G->mode[i];
            } else {
                double d = dist2(homex, homey, G->px[i], G->py[i]);
                tx[i] = homex; ty[i] = homey; spd[i] = TOP(i) * (d < 6 ? jog : 0.85);
                continue;
            }
        }
        if (md == JAB) {
            if (G->t >= G->muntil[i]) {
                G->mode[i] = md = CUT;
                G->mtx[i] = G->cex[i]; G->mty[i] = G->cey[i];
                G->muntil[i] = G->t + dist2(G->cex[i], G->cey[i], G->px[i], G->py[i]) / TOP(i) + 1.2;
            } else { tx[i] = G->mtx[i]; ty[i] = G->mty[i]; spd[i] = TOP(i) * 0.7; continue; }
        }
        if (md == CUT) {
            if (dist2(G->mtx[i], G->mty[i], G->px[i], G->py[i]) < 1.0 || G->t >= G->muntil[i]) {
                G->mode[i] = md = CLEAR;
                G->next_cut[i] = G->t + g->recut_delay[k];
            } else { tx[i] = G->mtx[i]; ty[i] = G->mty[i]; spd[i] = TOP(i); continue; }
        }
        if (md == CLEAR) {
            if (dist2(homex, homey, G->px[i], G->py[i]) < 2.0) G->mode[i] = SET;
            tx[i] = homex; ty[i] = homey; spd[i] = TOP(i) * 0.8;
        }
    }
}

/* ------------------------------------------------------------------ defense behaviour */
static void flight_pos(const Game *G, double s, double *x, double *y) {
    double o[2];
    us_flight_point(G->f.p0x, G->f.p0y, G->f.cx, G->f.cy, G->f.p1x, G->f.p1y, G->f.T, s, G->c->tail_time, o);
    *x = o[0]; *y = o[1];
}

static void enforce_double_team(Game *G, double *tx, double *ty, int mk, double *tvx, double *tvy) {
    if (G->holder < 0) return;
    double hx = G->px[G->holder], hy = G->py[G->holder];
    double r = G->c->double_team_radius + 0.2;
    for (int i = D0; i < D0 + 7; i++) {
        if (i == mk || G->mode[i] == CHASE) continue;
        int att = G->match[i];
        if (att >= 0 && dist2(G->px[att], G->py[att], hx, hy) < r) continue;
        double vx_ = tx[i] - hx, vy_ = ty[i] - hy, d = sqrt(vx_ * vx_ + vy_ * vy_);
        if (d < r) {
            if (d > 1e-6) { tx[i] = hx + vx_ / d * r; ty[i] = hy + vy_ / d * r; }
            else { tx[i] = hx + r; ty[i] = hy; }
            if (tvx) { tvx[i] = 0.0; tvy[i] = 0.0; }
        }
    }
}

/* Help defense: when a cutter has clearly beaten their defender, the nearest free defender (guarding a
 * stationary attacker away from the disc) steps into that cutter's throwing lane. */
static void help_on_cutter(Game *G, double *tx, double *ty, double *tvx, double *tvy, int mk, double rx, double ry,
                           const double *Pax, const double *Pay, const double *Vax, const double *Vay) {
    const UsDefense *g = DG;
    const UsConfig *c = G->c;
    if (g->help_cut <= 0.0 || G->holder < 0 || G->in_flight) return;
    int np_ = 0;
    for (int d = D0; d < D0 + 7; d++) np_ += G->match[d] >= 0;
    if (np_ < 2) return;
    int ba = -1, bd = -1;
    double bsep = 0.0;
    for (int d = D0; d < D0 + 7; d++) {
        int a = G->match[d];
        if (a < 0 || d == mk || a == G->holder || G->mode[d] == CHASE || G->mode[d] == GROUND) continue;
        int j = a - O0;
        double sep = dist2(Pax[j], Pay[j], G->px[d], G->py[d]);
        if (sqrt(sq(Vax[j]) + sq(Vay[j])) > 3.0 && sep > c->help_sep && (ba < 0 || sep > bsep)) { bsep = sep; ba = a; bd = d; }
    }
    if (ba < 0) return;
    int ja = ba - O0;
    double lx = Pax[ja] + 0.35 * (rx - Pax[ja]), ly = Pay[ja] + 0.35 * (ry - Pay[ja]);
    int h = -1;
    double hd = DBL_MAX;
    for (int d = D0; d < D0 + 7; d++) {
        int b = G->match[d];
        if (b < 0 || d == bd || d == mk || G->mode[d] == CHASE || G->mode[d] == GROUND) continue;
        int j = b - O0;
        if (!(sqrt(sq(Vax[j]) + sq(Vay[j])) < 3.0 && dist2(Pax[j], Pay[j], rx, ry) > 8.0)) continue;
        double dd = dist2(G->px[d], G->py[d], lx, ly);
        if (dd < hd) { hd = dd; h = d; }
    }
    if (h < 0) return;
    double nx = (1 - g->help_cut) * tx[h] + g->help_cut * lx, ny = (1 - g->help_cut) * ty[h] + g->help_cut * ly;
    clip_field(G, &nx, &ny, 0.3);
    tx[h] = nx; ty[h] = ny;
    if (tvx) { tvx[h] = 0.0; tvy[h] = 0.0; }
}

/* Switching (clam): every switch_interval, re-match person defenders to the nearest attackers if that saves
 * enough running; switch = 1 always re-matches, 0 never. */
static void maybe_switch(Game *G) {
    const UsDefense *g = DG;
    const UsConfig *c = G->c;
    if (g->switch_ <= 0.0 || G->t < G->next_switch) return;
    int np_ = 0;
    for (int d = D0; d < D0 + 7; d++) np_ += G->match[d] >= 0;
    if (np_ < 2) return;
    G->next_switch = G->t + c->switch_interval;
    int mk = marker_id(G), ds[7], att[7], n = 0;
    for (int d = D0; d < D0 + 7; d++)
        if (G->match[d] >= 0 && d != mk && G->mode[d] != CHASE && G->mode[d] != GROUND) { ds[n] = d; att[n] = G->match[d]; n++; }
    if (n < 2) return;
    double cost[49], cur = 0.0, opt = 0.0;
    int r2c[7];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cost[i * n + j] = dist2(G->px[ds[i]], G->py[ds[i]], G->px[att[j]], G->py[att[j]]);
    for (int i = 0; i < n; i++) cur += cost[i * n + i];
    us_hungarian(n, n, cost, r2c);
    for (int i = 0; i < n; i++) opt += cost[i * n + r2c[i]];
    if (cur - opt > (1.0 - g->switch_) * c->switch_gain)
        for (int i = 0; i < n; i++) { G->match[ds[i]] = att[r2c[i]]; G->def_of[att[r2c[i]]] = ds[i]; }
}

/* Person-D spot for an attacker at (pax,pay) with the disc at (rx,ry): cushion metres off the attacker, at the
 * genome's angle from the attacker->disc direction (0 = fronting, 180 = trailing). Returns the attacker-disc
 * distance. Shared by defender movement and by the thrower's read of the receiver's defender. */
static double person_spot(const Game *G, double pax, double pay, double rx, double ry, double *ox, double *oy) {
    const UsDefense *g = DG;
    double tdx = rx - pax, tdy = ry - pay, dist = sqrt(tdx * tdx + tdy * tdy);
    double ux = tdx / maxd(dist, 1e-6), uy = tdy / maxd(dist, 1e-6);
    double w = clampd((dist - 15.0) / 15.0, 0.0, 1.0);
    double th = ((1 - w) * g->angle_near + w * g->angle_far) * DEG2RAD;
    double dvx = cos(th) * ux, dvy = cos(th) * uy + sin(th) * G->s_open;
    double dn = maxd(sqrt(dvx * dvx + dvy * dvy), 1e-6);
    dvx /= dn; dvy /= dn;
    double cushion = g->cushion_base + g->cushion_per_m * dist;
    *ox = pax + cushion * dvx; *oy = pay + cushion * dvy;
    return dist;
}

static void defense_targets(Game *G, double *tx, double *ty, double *spd, double *tvx, double *tvy) {
    const UsDefense *g = DG;
    const UsConfig *c = G->c;
    double rx, ry;
    if (G->in_flight) {   /* positional defenders (zone and person) read the disc in flight late */
        double s = clampd((G->t - c->zone_read_delay - G->f.t0) / G->f.T, 0.0, G->f.s_end);
        flight_pos(G, s, &rx, &ry);
    } else { rx = G->dx; ry = G->dy; }
    int mk = marker_id(G);
    int chasing[NP];
    for (int i = 0; i < NP; i++) chasing[i] = G->mode[i] == CHASE;
    double zx[7], zy[7], Pax[7], Pay[7], Vax[7], Vay[7];
    for (int j = 0; j < 7; j++) delayed(G, O0 + j, 0.25, &Pax[j], &Pay[j], &Vax[j], &Vay[j]);
    int any_zone = 0;
    for (int i = D0; i < D0 + 7; i++) any_zone |= G->zslot[i] >= 0;
    if (any_zone) zone_points(G, rx, ry, zx, zy);
    for (int i = D0; i < D0 + 7; i++) {
        if (G->t < G->ground_until[i]) { tx[i] = G->px[i]; ty[i] = G->py[i]; spd[i] = 0.0; continue; }
        if (G->mode[i] == GROUND) G->mode[i] = rest_mode(G, i);
        if (chasing[i]) { tx[i] = G->mtx[i]; ty[i] = G->mty[i]; spd[i] = TOP(i); continue; }
        if (i == mk) {
            double mc, mw;
            mark_geometry(G, &mc, &mw);
            tx[i] = G->px[G->holder] + g->mark_dist * cos(mc * DEG2RAD);
            ty[i] = G->py[G->holder] + g->mark_dist * sin(mc * DEG2RAD);
            spd[i] = TOP(i);
            continue;
        }
        if (G->zslot[i] >= 0 || G->match[i] < 0) {
            double pxz = G->zslot[i] >= 0 ? zx[G->zslot[i]] : G->px[i];
            double pyz = G->zslot[i] >= 0 ? zy[G->zslot[i]] : G->py[i];
            int nn = 0; double bd = DBL_MAX;
            for (int j = 0; j < 7; j++) { double d = dist2(Pax[j], Pay[j], pxz, pyz); if (d < bd) { bd = d; nn = j; } }
            if (bd < 8.0) {
                double shx = g->zone_shade * (Pax[nn] - pxz), shy = g->zone_shade * (Pay[nn] - pyz);
                double ln = sqrt(shx * shx + shy * shy);
                if (ln > 5.0) { shx *= 5.0 / ln; shy *= 5.0 / ln; }
                pxz += shx; pyz += shy;
            }
            clip_field(G, &pxz, &pyz, 0.8);
            tx[i] = pxz; ty[i] = pyz; spd[i] = TOP(i);
            continue;
        }
        double Pdx, Pdy, Vdx, Vdy;
        double rt = G->P[i].reaction_time;
        double blind = 0.0;   /* fronting: the attacker is behind the defender, so they react later */
        {
            int a = G->match[i];
            double u1x = G->px[i] - G->px[a], u1y = G->py[i] - G->py[a], u2x = rx - G->px[a], u2y = ry - G->py[a];
            double n1 = sqrt(u1x * u1x + u1y * u1y), n2 = sqrt(u2x * u2x + u2y * u2y);
            if (n1 > 1e-6 && n2 > 1e-6) blind = c->blind_delay * maxd(0.0, (u1x * u2x + u1y * u2y) / (n1 * n2));
        }
        delayed(G, G->match[i], rt + blind, &Pdx, &Pdy, &Vdx, &Vdy);
        double pax = Pdx + g->anticipation * Vdx * rt, pay = Pdy + g->anticipation * Vdy * rt;
        double ppx, ppy, dist = person_spot(G, pax, pay, rx, ry, &ppx, &ppy);
        int slow = sqrt(Vdx * Vdx + Vdy * Vdy) < 3.0;
        if (slow && dist > 12.0) {
            ppx = (1 - g->poach) * ppx + g->poach * (rx + 0.55 * (pax - rx));
            ppy = (1 - g->poach) * ppy + g->poach * (ry + 0.55 * (pay - ry));
        }
        if (slow && g->stack_sag > 0.0) {   /* surround the stack */
            int ai = G->match[i] - O0, cnt = 0;
            double cxs = 0.0, cys = 0.0;
            for (int j = 0; j < 7; j++)
                if (dist2(Pax[j], Pay[j], Pax[ai], Pay[ai]) < c->stack_radius) { cxs += Pax[j]; cys += Pay[j]; cnt++; }
            if (cnt >= 3) {
                cxs /= cnt; cys /= cnt;
                double tcx = rx - cxs, tcy = ry - cys, tn = maxd(sqrt(tcx * tcx + tcy * tcy), 1e-6);
                double sgx = cxs + 2.0 * tcx / tn, sgy = cys + 2.0 * tcy / tn, ws = 0.6 * g->stack_sag;
                ppx = (1 - ws) * ppx + ws * sgx;
                ppy = (1 - ws) * ppy + ws * sgy;
            }
        }
        clip_field(G, &ppx, &ppy, 0.3);
        tx[i] = ppx; ty[i] = ppy; spd[i] = TOP(i);
        if (tvx) { tvx[i] = Vdx; tvy[i] = Vdy; }
    }
    help_on_cutter(G, tx, ty, tvx, tvy, mk, rx, ry, Pax, Pay, Vax, Vay);
    enforce_double_team(G, tx, ty, mk, tvx, tvy);
}

/* ------------------------------------------------------------------ value field */
static double value_at(const Game *G, double x, double y) {
    const UsConfig *c = G->c;
    if (x >= c->L - c->ez && x <= c->L && y >= 0 && y <= c->W) return 1.0;
    double u = clampd((x - c->ez) / (c->L - 2 * c->ez), 0.0, 1.0);
    double v = c->v_base + c->v_gain * pow(u, c->v_power);
    double edge = mind(y, c->W - y);
    return v - c->v_edge * clampd(1.0 - edge / 5.0, 0.0, 1.0);
}
static double turnover_cost(const Game *G, double x, double y) {
    const UsConfig *c = G->c;
    double mx = clampd(c->L - x, c->ez, c->L - c->ez - 0.01), my = c->W - y;
    return value_at(G, mx, my);
}

/* ------------------------------------------------------------------ throw evaluation */
typedef struct {
    int n;
    int ttype[MAXC], curve[MAXC], rcv[MAXC], huck[MAXC], is_break[MAXC], loft[MAXC];
    double tx[MAXC], ty[MAXC], rpx[MAXC], rpy[MAXC], dist[MAXC], T[MAXC], sig[MAXC];
    double p[MAXC], p_mark[MAXC], U[MAXC];
    int marked;
} Cands;

static void predict_receiver(const Game *G, int i, double tau, double *ox, double *oy) {
    double px = G->px[i], py = G->py[i], vx = G->vx[i], vy = G->vy[i];
    double vm = sqrt(vx * vx + vy * vy);
    if (G->mode[i] == CUT || G->mode[i] == JAB || G->mode[i] == RESET) {
        double gx = G->mode[i] != JAB ? G->mtx[i] : G->cex[i], gy = G->mode[i] != JAB ? G->mty[i] : G->cey[i];
        double dgx = gx - px, dgy = gy - py, ln = sqrt(dgx * dgx + dgy * dgy);
        double travel = mind(mind(vm * tau + 0.5 * G->P[i].accel * tau * tau * 0.5, TOP(i) * tau), ln + 3.0);
        *ox = px + dgx / maxd(ln, 1e-6) * travel;
        *oy = py + dgy / maxd(ln, 1e-6) * travel;
    } else {
        double gx = G->mode[i] == CLEAR ? G->mtx[i] : px, gy = G->mode[i] == CLEAR ? G->mty[i] : py;
        double travel = mind(vm * tau, dist2(gx, gy, px, py) + 1.0);
        *ox = px + vx / maxd(vm, 1e-6) * travel;
        *oy = py + vy / maxd(vm, 1e-6) * travel;
    }
}

/* Fast exp (rel. error ~1e-7) written so loops calling it auto-vectorize. */
static inline double fast_exp(double x) {
    x = clampd(x, -700.0, 700.0);
    double t = x * 1.4426950408889634;           /* x / ln 2 */
    double n = floor(t + 0.5);
    double f = (t - n) * 0.6931471805599453;     /* remainder, |f| <= ln2/2 */
    double p = 1.0 + f * (1.0 + f * (0.5 + f * (1.0 / 6 + f * (1.0 / 24 + f * (1.0 / 120 + f * (1.0 / 720 + f * (1.0 / 5040)))))));
    union { double d; int64_t i; } u;
    u.i = (int64_t)(n + 1023.0) << 52;
    return p * u.d;
}
static inline double fast_sigm(double x) { return 1.0 / (1.0 + fast_exp(-x)); }
/* Phi(x); returns exactly 1.0 beyond 8.3 sd, where erfc already underflows to that in double precision. */
static inline double ndtr_far(double x) { return x > 8.3 ? 1.0 : ndtr_(x); }

/* Defender vs. one candidate flight: arrival time (incl. reaction delay dl, lane-standing and huck read)
 * and reflex factor at each of the KS path samples. Branch-free so the k-loop vectorizes; identical
 * maths to us_time_to_reach + the reference's in-lane rule. */
static void defender_kernel(double px, double py, double vx, double vy, double top, double accel, double brake,
                            const double *reach, const double *alw, double dl, const double *ptx, const double *pty, const double *tk,
                            const double *lane, double reflex_block, double huck_extra, double *tdef, double *reflex) {
    const double prx = px + vx * dl, pry = py + vy * dl;
    const double inv2a = 1.0 / (2.0 * accel), inva = 1.0 / accel, invb = 1.0 / brake, invtop = 1.0 / top;
    for (int k = 0; k < KS; k++) {
        double dx = ptx[k] - prx, dy = pty[k] - pry;
        double d = sqrt(dx * dx + dy * dy);
        double d_eff = d - reach[k] - lane[k];
        double inv = 1.0 / (d > 1e-9 ? d : 1e-9);
        double v_par = (vx * dx + vy * dy) * inv;
        double v_perp = fabs(vx * dy - vy * dx) * inv;
        double t_turn = ((-v_par > 0.0 ? -v_par : 0.0) + 0.5 * v_perp) * invb;
        double v0 = v_par > 0.0 ? v_par : 0.0;
        double d_acc = (top * top - v0 * v0) * inv2a;
        double de = d_eff > 0.0 ? d_eff : 0.0;
        double t_short = (-v0 + sqrt(v0 * v0 + 2.0 * accel * de)) * inva;
        double t_long = (top - v0) * inva + (de - d_acc) * invtop;
        double t_run = de <= d_acc ? t_short : t_long;
        double td = (d_eff <= 0.0 ? 0.0 : t_turn + t_run) + dl;
        double tm = tk[k] < dl ? tk[k] : dl;
        double rr = reach[k] + lane[k];
        rr *= rr;
        double ax0 = px - ptx[k], ay0 = py - pty[k];
        double ax1 = ax0 + vx * 0.5 * tm, ay1 = ay0 + vy * 0.5 * tm;
        double ax2 = ax0 + vx * tm, ay2 = ay0 + vy * tm;
        int in_lane = ((ax0 * ax0 + ay0 * ay0 <= rr) | (ax1 * ax1 + ay1 * ay1 <= rr) | (ax2 * ax2 + ay2 * ay2 <= rr))
                      & (alw[k] > 0.5);
        double tl = tk[k] - 0.3;
        td = in_lane && tl < td ? tl : td;
        reflex[k] = (in_lane && tk[k] < dl) ? reflex_block : 1.0;
        tdef[k] = alw[k] > 0.5 ? td + huck_extra : 1e9;   /* disc above this defender's reach there */
    }
}

static int evaluate_throws(Game *G, Cands *C) {
    const UsConfig *c = G->c;
    const UsOffense *g = OG;
    const int h = G->holder;
    const double hpx = G->px[h], hpy = G->py[h];
    int R[7], nR = 0;
    for (int i = O0; i < O0 + 7; i++) if (i != h && G->t >= G->ground_until[i]) R[nR++] = i;
    if (nR == 0) return 0;
    static const int menu_t[NMENU] = {BH, BH, BH, BH, FH, FH, FH, FH, HAM};
    static const int menu_c[NMENU] = {0, 0, 1, -1, 0, 0, 1, -1, 0};
    static const int menu_l[NMENU] = {0, 1, 0, 0, 0, 1, 0, 0, 0};   /* lofted variant of the straight throws */
    /* receiver targets per lead */
    double tgx[14], tgy[14];
    int tgr[14], nT = 0;
    for (int li = 0; li < 2; li++) {
        double lead = li == 0 ? 0.0 : g->lead;
        for (int j = 0; j < nR; j++) {
            int r = R[j];
            double tau = 0.5, tpx = 0, tpy = 0;
            for (int it = 0; it < 2; it++) {
                predict_receiver(G, r, tau, &tpx, &tpy);
                tau = c->bh_hang + dist2(tpx, tpy, hpx, hpy) / c->bh_speed;
            }
            double vm = sqrt(sq(G->vx[r]) + sq(G->vy[r]));
            if (vm > 2.0) { tpx += G->vx[r] / maxd(vm, 1e-6) * lead; tpy += G->vy[r] / maxd(vm, 1e-6) * lead; }
            tgx[nT] = tpx; tgy[nT] = tpy; tgr[nT] = r; nT++;
        }
    }
    /* marker geometry (same for every candidate) */
    int marked = marker_set(G), mk = marker_id(G);
    double mc = 0, mw = 0;
    if (marked) mark_geometry(G, &mc, &mw);
    int fake_active = G->t < G->fake_effect_until;
    int def_of[NP];
    for (int i = 0; i < NP; i++) def_of[i] = G->def_of[i];
    double base_dl[7], help_dl = c->help_read_delay;   /* applied per defender: person role, not on the receiver */
    for (int j = 0; j < 7; j++) base_dl[j] = G->P[D0 + j].reaction_time + c->disc_read_delay;
    const double bp = c->block_prob;
    const double noise_sd = c->read_noise_base + c->read_noise_skill * G->P[h].decision_noise;
    const double urg = urgency(G);

    /* break-side targets (same test as is_break): curves and hammers are only offered there when
     * specials_everywhere is off */
    int brk_pre[14];
    for (int ti = 0; ti < nT; ti++)
        brk_pre[ti] = marked && fabs(wrap_deg(atan2(tgy[ti] - hpy, tgx[ti] - hpx) * RAD2DEG - mc)) < c->break_side_deg;
    int n = 0;
    int idx_of[14][NMENU];
    double Vs[MAXC], Cs[MAXC];
    for (int ti = 0; ti < nT; ti++) {
        /* a non-moving receiver gets the same target with and without lead: reuse, redraw only the noise */
        int dup = (ti >= nR && tgx[ti] == tgx[ti - nR] && tgy[ti] == tgy[ti - nR]) ? ti - nR : -1;
        for (int m = 0; m < NMENU; m++) {
            idx_of[ti][m] = -1;
            if (dup >= 0) {
                int q = idx_of[dup][m];
                if (q < 0) continue;
                double pq = C->p[q];
                double p_hat = sigm(log(pq / (1 - pq)) + rng_n(&G->rng) * noise_sd);
                C->ttype[n] = C->ttype[q]; C->curve[n] = C->curve[q]; C->rcv[n] = C->rcv[q]; C->huck[n] = C->huck[q];
                C->loft[n] = C->loft[q];
                C->is_break[n] = C->is_break[q]; C->tx[n] = C->tx[q]; C->ty[n] = C->ty[q]; C->rpx[n] = C->rpx[q];
                C->rpy[n] = C->rpy[q]; C->dist[n] = C->dist[q]; C->T[n] = C->T[q]; C->sig[n] = C->sig[q];
                C->p[n] = pq; C->p_mark[n] = C->p_mark[q]; Vs[n] = Vs[q]; Cs[n] = Cs[q];
                C->U[n] = p_hat * Vs[q] - (1 - p_hat) * g->risk_aversion * urg * Cs[q] + g->huck_bias * C->huck[q]
                          + g->break_bias * C->is_break[q] + g->hammer_bias * (C->ttype[q] == HAM);
                idx_of[ti][m] = n;
                n++;
                continue;
            }
            int tt = menu_t[m], cv = menu_c[m], lf = menu_l[m], r = tgr[ti];
            double rpx = hpx, rpy = hpy + type_side(c, tt) * 1.0;
            double tx = tgx[ti], ty = tgy[ti];
            double dist = dist2(tx, ty, rpx, rpy);
            int huck = dist > c->huck_min_dist && tt != HAM;
            double range = tt == BH ? G->P[h].range_backhand : (tt == FH ? G->P[h].range_forehand : G->P[h].range_hammer);
            if (!(dist >= c->min_throw_dist && dist <= range)) continue;
            if (tt == HAM && dist < 10.0) continue;
            if (cv != 0 && dist < 8.0) continue;
            if (lf && (huck || dist < c->loft_min_dist || dist > c->loft_max_dist)) continue;
            if (c->specials_everywhere < 0.5 && (cv != 0 || tt == HAM) && !brk_pre[ti]) continue;
            double T = flight_time(c, tt, dist, huck, lf);
            double peak = throw_peak(c, tt, dist, huck, lf);
            double cx, cy;
            control_point(c, rpx, rpy, tx, ty, (double)cv, &cx, &cy);
            double ptx[KS], pty[KS], sk[KS], tk[KS];
            int reach_k[KS];
            double hk_[KS], low_[KS];
            for (int k = 0; k < KS; k++) {
                sk[k] = 0.1 + 0.1 * k;
                bezier(rpx, rpy, cx, cy, tx, ty, sk[k], &ptx[k], &pty[k]);
                tk[k] = sk[k] * T;
                hk_[k] = flight_height(c, peak, sk[k], T, tt == HAM ? c->hammer_release : c->h_release);
                low_[k] = hk_[k] < c->low_disc ? c->low_mult : 1.0;
                reach_k[k] = sk[k] * dist >= c->double_team_radius;
            }
            /* mark */
            int is_break = 0;
            double p_mark = 0.0;
            if (marked) {
                double ang = atan2(ty - hpy, tx - hpx) * RAD2DEG;
                is_break = fabs(wrap_deg(ang - mc)) < c->break_side_deg;   /* toward the marker's side */
                double sm = mind(1.0, 2.0 / maxd(dist, 1e-6)), ex, ey;
                bezier(rpx, rpy, cx, cy, tx, ty, sm, &ex, &ey);
                {   /* bisector of release direction and target direction (curves can't launder a break) */
                    double u1x = ex - hpx, u1y = ey - hpy, n1 = maxd(sqrt(u1x * u1x + u1y * u1y), 1e-9);
                    double u2x = tx - hpx, u2y = ty - hpy, n2 = maxd(sqrt(u2x * u2x + u2y * u2y), 1e-9);
                    double ubx = u1x / n1 + u2x / n2, uby = u1y / n1 + u2y / n2, nb = maxd(sqrt(ubx * ubx + uby * uby), 1e-9);
                    ex = hpx + 2.0 * ubx / nb; ey = hpy + 2.0 * uby / nb;
                }
                p_mark = us_mark_block_prob(c, hpx, hpy, ex, ey, tt, mc, mw, G->P[mk].marking, G->P[h].break_skill, fake_active)
                         * mark_quality(G);
            }
            double sig = us_error_sigma(c, G->P[h].throw_skill, tt, dist, is_break, marked, cv != 0);
            double lane[KS];
            for (int k = 0; k < KS; k++) lane[k] = c->model_error_margin * sig * sk[k];
            /* receiver */
            double trec = us_time_to_reach(G->px[r], G->py[r], G->vx[r], G->vy[r], TOP(r), G->P[r].accel,
                                           G->brake[r], G->reach[r], tx, ty, 0.0, 0.0);
            /* defenders. Bounding circle of the path (inside the AABB of release/control/target) lets us
             * skip defenders who cannot get within 2 s of the disc anywhere: the learned model clips time
             * margins to +-2 s and their D chance is < 1e-7, so the result is unchanged. */
            double bx0 = mind(mind(rpx, cx), tx), bx1 = maxd(maxd(rpx, cx), tx);
            double by0 = mind(mind(rpy, cy), ty), by1 = maxd(maxd(rpy, cy), ty);
            double bcx = 0.5 * (bx0 + bx1), bcy = 0.5 * (by0 + by1), brad = 0.5 * sqrt(sq(bx1 - bx0) + sq(by1 - by0));
            double tdef[7][KS], reflex[7][KS];
            int skip[7];
            for (int j = 0; j < 7; j++) {
                int d = D0 + j;
                double dl = base_dl[j] + ((G->match[d] >= 0 && d != def_of[r]) ? help_dl : 0.0);
                double vm = sqrt(sq(G->vx[d]) + sq(G->vy[d]));
                double gap_lb = dist2(G->px[d], G->py[d], bcx, bcy) - brad - vm * dl - G->reach[d] - lane[KS - 1];
                skip[j] = d != def_of[r] && gap_lb > 0.0 && dl + gap_lb / TOP(d) + c->huck_read_extra * huck - T > 2.0;
                if (skip[j]) {
                    for (int k = 0; k < KS; k++) { tdef[j][k] = 1e9; reflex[j][k] = 1.0; }
                    continue;
                }
                double rk[KS], alw[KS];   /* reach envelope along this flight for this defender */
                for (int k = 0; k < KS; k++) {
                    alw[k] = hk_[k] <= G->maxreach[d] ? 1.0 : 0.0;
                    rk[k] = G->reach[d] * (hk_[k] > G->standing[d] ? c->jump_reach_mult : 1.0);
                }
                defender_kernel(G->px[d], G->py[d], G->vx[d], G->vy[d], TOP(d), G->P[d].accel, G->brake[d],
                                rk, alw, dl, ptx, pty, tk, lane, c->reflex_block, c->huck_read_extra * huck,
                                tdef[j], reflex[j]);
            }
            if (def_of[r] >= 0 && G->zslot[def_of[r]] < 0) {
                /* the receiver's person defender doesn't drift: they move with the receiver toward their spot
                 * relative to the catch point. A fronting defender is still in the lane when the disc arrives. */
                int d = def_of[r], j = d - D0;
                double rt = G->P[d].reaction_time, inv_tr = 1.0 / maxd(trec, 0.05), hx, hy, s0x, s0y;
                person_spot(G, G->px[r], G->py[r], hpx, hpy, &s0x, &s0y);
                person_spot(G, tx, ty, hpx, hpy, &hx, &hy);
                double ex = G->px[d] - s0x, ey = G->py[d] - s0y, e0 = maxd(sqrt(ex * ex + ey * ey), 1e-6);
                for (int k = 0; k < KS; k++) {
                    if (hk_[k] > G->maxreach[d]) continue;
                    double fr = clampd((tk[k] - rt) * inv_tr, 0.0, 1.0);
                    double cl = clampd(1.0 - maxd(tk[k] - rt, 0.0) * TOP(d) / e0, 0.0, 1.0);
                    double qx = s0x + (hx - s0x) * fr + ex * cl - ptx[k], qy = s0y + (hy - s0y) * fr + ey * cl - pty[k];
                    double rr = G->reach[d] * (hk_[k] > G->standing[d] ? c->jump_reach_mult : 1.0) + lane[k];
                    if (qx * qx + qy * qy <= rr * rr) {
                        tdef[j][k] = mind(tdef[j][k], tk[k] - 0.3 + c->huck_read_extra * huck);
                        if (tk[k] < base_dl[j]) reflex[j][k] = c->reflex_block;
                    }
                }
            }
            if (def_of[r] >= 0) {
                int d = def_of[r], j = d - D0;
                double gap =maxd(dist2(G->px[d], G->py[d], G->px[r], G->py[r]) - G->reach[d] - lane[KS - 1], 0.0);
                double react = G->P[d].reaction_time + c->disc_read_delay;
                double track = maxd(trec + 0.5 * G->P[d].reaction_time, react) + gap / TOP(d);
                tdef[j][KS - 1] = mind(tdef[j][KS - 1], track);
            }
            double p_def_keep = 1.0, mid_margin = 9.0, end_margin = DBL_MAX, mtr = mind(trec, T);
            for (int j = 0; j < 7; j++) {
                if (skip[j]) continue;
                double pmid = 0.0, mm = 9.0;
                const double *tdj = tdef[j], *rfj = reflex[j];
                for (int k = 0; k < KS - 1; k++) {
                    if (!reach_k[k]) continue;
                    double gap = tdj[k] - tk[k];
                    mm = gap < mm ? gap : mm;
                    /* > 2 s behind the disc: sigmoid < 6e-8, same cut-off as the defender culling */
                    if (gap < 2.0) {
                        double v = fast_sigm(-gap * (1.0 / 0.12)) * bp * rfj[k] * low_[k];
                        pmid = v > pmid ? v : pmid;
                    }
                }
                if (mm < mid_margin) mid_margin = mm;
                double te = tdef[j][KS - 1];
                double pend = (reach_k[KS - 1] && te - T < 2.0)
                                  ? fast_sigm((T - te) / 0.15) * fast_sigm((mtr - te + 0.1) / 0.12) * bp * 0.8 : 0.0;
                p_def_keep *= 1.0 - maxd(pmid, pend);
                if (te - mtr < end_margin) end_margin = te - mtr;
            }
            double p_def = 1.0 - p_def_keep;
            double p_recv = fast_sigm((T + 0.15 - trec) / 0.15);
            double radius = G->reach[r] + 0.3 + c->receiver_adjust * maxd(0.0, T - trec);
            double p_ok = 1.0 - fast_exp(-radius * radius / (2 * sig * sig));
            double p_in = ndtr_far(ty / sig) * ndtr_far((c->W - ty) / sig) * ndtr_far((c->L - tx) / sig);
            double vdisc = dist / T;
            int near_def = 0;
            for (int j = D0; j < D0 + 7 && !near_def; j++) near_def = dist2(G->px[j], G->py[j], tx, ty) < 2.0;
            double hands = G->P[r].hands - c->speed_drop_per_ms * maxd(vdisc - 12.0, 0.0) - c->contest_drop * near_def
                           - c->huck_drop * huck - c->hammer_drop * (tt == HAM);
            double p = (1 - p_mark) * (1 - p_def) * p_recv * p_ok * p_in * clampd(hands, 0.0, 1.0);
            p = clampd(p, 1e-4, 1 - 1e-4);
            if (c->use_model > 0.5) {
                double mm = clampd(mid_margin, -2.0, 2.0), em = clampd(end_margin, -2.0, 2.0);
                double X[US_N_MODEL_W] = {
                    1.0, log(p / (1 - p)), log(clampd(1 - p_def, 1e-4, 1.0)), log(clampd(p_recv, 1e-4, 1.0)),
                    log(clampd(p_ok, 1e-4, 1.0)), log(clampd(1 - p_mark, 1e-4, 1.0)), log(clampd(p_in, 1e-4, 1.0)),
                    clampd(hands, 0.0, 1.0), mm, em, mind(mm, 0.0), mind(em, 0.0), dist / 30.0, sq(dist / 30.0), T,
                    (double)huck, (double)(tt == HAM), (double)(cv != 0), (double)is_break,
                    G->stall / c->stall_max, sqrt(sq(G->vx[r]) + sq(G->vy[r])) / 8.0, clampd(trec - T, -2.0, 2.0),
                    (double)lf};
                double z = 0.0;
                for (int q = 0; q < US_N_MODEL_W; q++) z += X[q] * G->model_w[q];
                p = clampd(fast_sigm(z), 1e-4, 1 - 1e-4);
            }
            double p_hat = fast_sigm(log(p / (1 - p)) + rng_n(&G->rng) * noise_sd);
            double V = value_at(G, tx, ty), Cst = turnover_cost(G, tx, ty);
            double U = p_hat * V - (1 - p_hat) * g->risk_aversion * urg * Cst
                       + g->huck_bias * huck + g->break_bias * is_break + g->hammer_bias * (tt == HAM);
            C->ttype[n] = tt; C->curve[n] = cv; C->rcv[n] = r; C->huck[n] = huck; C->is_break[n] = is_break;
            C->tx[n] = tx; C->ty[n] = ty; C->rpx[n] = rpx; C->rpy[n] = rpy; C->dist[n] = dist; C->T[n] = T;
            C->sig[n] = sig; C->p[n] = p; C->p_mark[n] = p_mark; C->U[n] = U; C->loft[n] = lf;
            Vs[n] = V; Cs[n] = Cst;
            idx_of[ti][m] = n;
            n++;
        }
    }
    C->n = n;
    C->marked = marked;
    return n;
}

/* ------------------------------------------------------------------ possession end / events */
static void end_possession(Game *G, int outcome, double x, double y) {
    G->res.outcome = outcome;
    G->res.duration = G->t - G->poss_t0;
    G->res.end_x = x; G->res.end_y = y;
    G->in_flight = 0;
    G->holder = -1;
}

static void do_fake(Game *G) {
    const UsConfig *c = G->c;
    G->last_fake = G->t;
    G->fake_until = G->t + c->fake_time;
    int mk = marker_id(G);
    double dur = mk >= 0 ? c->fake_effect_time * (1.2 - 0.6 * G->P[mk].marking) : 0.0;
    G->fake_effect_until = G->fake_until + dur;
    G->next_decision = G->fake_until;
    G->res.fakes++;
}

static void release(Game *G, const Cands *C, int b) {
    const UsConfig *c = G->c;
    const int h = G->holder;
    int tt = C->ttype[b], cv = C->curve[b], huck = C->huck[b], isb = C->is_break[b], r = C->rcv[b], lf = C->loft[b];
    double ix = C->tx[b], iy = C->ty[b], rpx = C->rpx[b], rpy = C->rpy[b];
    G->res.throws++;
    G->res.hucks += huck;
    G->res.breaks += isb;
    G->res.hammers += tt == HAM;
    G->res.sum_stall += G->stall;
    G->res.sum_p += C->p[b];
    if (C->marked && rng_u(&G->rng) < C->p_mark[b]) {
        double dl = maxd(dist2(ix, iy, rpx, rpy), 1e-6);
        double sx = G->px[h] + (ix - rpx) / dl * 1.5, sy = G->py[h] + (iy - rpy) / dl * 1.5;
        clip_field(G, &sx, &sy, 0.8);
        end_possession(G, US_MARK_BLOCK, sx, sy);
        return;
    }
    double ex = rng_n(&G->rng) * C->sig[b], ey = rng_n(&G->rng) * C->sig[b];
    double p1x = ix + ex, p1y = iy + ey;
    double dist = dist2(p1x, p1y, rpx, rpy);
    double T = flight_time(c, tt, dist, huck, lf);
    Flight *f = &G->f;
    memset(f, 0, sizeof *f);
    f->thrower = h; f->receiver = r; f->ttype = tt; f->huck = huck; f->curve = cv; f->is_break = isb;
    f->p0x = rpx; f->p0y = rpy; f->p1x = p1x; f->p1y = p1y; f->ix = ix; f->iy = iy;
    control_point(c, rpx, rpy, p1x, p1y, (double)cv, &f->cx, &f->cy);
    f->t0 = G->t; f->T = T; f->dist = dist;
    f->peak = throw_peak(c, tt, dist, huck, lf);
    f->h0 = tt == HAM ? c->hammer_release : c->h_release;   /* hammers go overhead */
    f->loft = lf;
    f->s_end = 1.0 + c->tail_time / T;
    f->prevx = rpx; f->prevy = rpy;
    G->in_flight = 1;
    G->holder = -1;
    G->stall = 0.0;
    G->reset_player = -1;
    const UsOffense *g = OG;
    if (rng_u(&G->rng) < g->give_go) {
        double gx = G->px[h] + cos(15.0 * DEG2RAD) * 10.0, gy = G->py[h] + G->s_open * sin(15.0 * DEG2RAD) * 10.0;
        clip_field(G, &gx, &gy, 0.8);
        G->mode[h] = CUT; G->mtx[h] = gx; G->mty[h] = gy; G->muntil[h] = G->t + 2.0;
    } else G->mode[h] = CLEAR;
    for (int i = O0; i < O0 + 7; i++) if (G->mode[i] == RESET) G->mode[i] = CLEAR;
    G->mode[r] = CHASE; G->mtx[r] = ix; G->mty[r] = iy;
    int others[NSLOT], n = 0;
    for (int i = O0; i < O0 + 7; i++) if (i != r) others[n++] = i;
    assign_slots(G, others, ix, iy);
}

static void thrower_decision(Game *G) {
    const UsOffense *g = OG;
    const UsConfig *c = G->c;
    Cands C;
    if (!evaluate_throws(G, &C)) return;
    int b = 0;
    for (int q = 1; q < C.n; q++) if (C.U[q] > C.U[b]) b = q;
    if (c->explore_eps > 0 && rng_u(&G->rng) < c->explore_eps) {
        int top[12], nt = 0, used[MAXC] = {0};
        for (; nt < 12 && nt < C.n; nt++) {
            int bb = -1;
            for (int q = 0; q < C.n; q++) if (!used[q] && (bb < 0 || C.U[q] > C.U[bb])) bb = q;
            used[bb] = 1; top[nt] = bb;
        }
        b = top[(int)(rng_u(&G->rng) * nt)];
        C.U[b] = DBL_MAX;
    }
    double here = value_at(G, G->px[G->holder], G->py[G->holder]);
    double s_eff = maxd(G->stall, G->t - G->ready_time);
    double hold = here * g->patience * urgency(G) * maxd(1.0 - sq(s_eff / c->stall_max), 0.0);
    int must = s_eff >= g->panic_stall;
    if (C.U[b] < hold && !must) {
        if (C.is_break[b] && C.p_mark[b] > 0.12 && G->t - G->last_fake > 1.2 &&
            rng_u(&G->rng) < g->fake_rate * c->decision_interval * 2) do_fake(G);
        return;
    }
    if (C.is_break[b] && C.p_mark[b] > 0.2 && G->t - G->last_fake > 1.2 && !must && rng_u(&G->rng) < g->fake_rate) {
        do_fake(G);
        return;
    }
    release(G, &C, b);
}

/* ------------------------------------------------------------------ disc in the air */
/* Arrival times of one player at n points: us_time_to_reach (no delay/extra) as a branch-free loop. */
static void ttr_points(double px, double py, double vx, double vy, double top, double accel, double brake,
                       double reach, const double *qx, const double *qy, int n, double *out) {
    const double inv2a = 1.0 / (2.0 * accel), inva = 1.0 / accel, invb = 1.0 / brake, invtop = 1.0 / top;
    for (int k = 0; k < n; k++) {
        double dx = qx[k] - px, dy = qy[k] - py;
        double d = sqrt(dx * dx + dy * dy);
        double d_eff = d - reach;
        double inv = 1.0 / (d > 1e-9 ? d : 1e-9);
        double v_par = (vx * dx + vy * dy) * inv;
        double v_perp = fabs(vx * dy - vy * dx) * inv;
        double t_turn = ((-v_par > 0.0 ? -v_par : 0.0) + 0.5 * v_perp) * invb;
        double v0 = v_par > 0.0 ? v_par : 0.0;
        double d_acc = (top * top - v0 * v0) * inv2a;
        double de = d_eff > 0.0 ? d_eff : 0.0;
        double t_short = (-v0 + sqrt(v0 * v0 + 2.0 * accel * de)) * inva;
        double t_long = (top - v0) * inva + (de - d_acc) * invtop;
        double t_run = de <= d_acc ? t_short : t_long;
        out[k] = d_eff <= 0.0 ? 0.0 : t_turn + t_run;
    }
}

static void update_chasers(Game *G) {
    const UsConfig *c = G->c;
    const Flight *f = &G->f;
    double since = G->t - f->t0, s_now = since / f->T;
    double ss[KCH], ptx[KCH], pty[KCH], td[KCH], hss[KCH];
    int ok[KCH];
    double s_a = mind(s_now + 0.02, f->s_end);
    for (int k = 0; k < KCH; k++) {
        ss[k] = s_a + (f->s_end - s_a) * k / (KCH - 1);
        flight_pos(G, ss[k], &ptx[k], &pty[k]);
        td[k] = ss[k] * f->T - since;
        ok[k] = ss[k] * f->dist >= c->double_team_radius;
        hss[k] = flight_height(c, f->peak, ss[k], f->T, f->h0);
    }
    ok[KCH - 1] = 1;
    double read_dl = c->disc_read_delay + (f->huck ? c->huck_read_extra : 0.0);
    int read = since >= read_dl;
    int r = f->receiver;
    if (G->mode[r] == CHASE) {
        if (read) {
            int got = -1;
            double tr[KCH];
            ttr_points(G->px[r], G->py[r], G->vx[r], G->vy[r], TOP(r), G->P[r].accel, G->brake[r], G->reach[r],
                       ptx, pty, KCH, tr);
            for (int k = 0; k < KCH; k++)
                if (ok[k] && (hss[k] <= G->maxreach[r] || k == KCH - 1) && tr[k] <= td[k]) { got = k; break; }
            if (got < 0) got = KCH - 1;
            G->mtx[r] = ptx[got]; G->mty[r] = pty[got];
        } else { G->mtx[r] = f->ix; G->mty[r] = f->iy; }
    }
    if (!read) return;
    const UsDefense *g = DG;
    int rd_def = G->def_of[r];
    for (int i = D0; i < D0 + 7; i++) {
        if (!(G->t >= G->ground_until[i] && since >= G->P[i].reaction_time + read_dl + help_delay(G, i, r))) continue;
        double margin = g->help_margin + (i == rd_def ? 0.6 : 0.0);
        int got = -1;
        double tdd[KCH];
        ttr_points(G->px[i], G->py[i], G->vx[i], G->vy[i], TOP(i), G->P[i].accel, G->brake[i], G->reach[i],
                   ptx, pty, KCH, tdd);
        for (int k = 0; k < KCH; k++)
            if (ok[k] && (hss[k] <= G->maxreach[i] || k == KCH - 1) && tdd[k] <= td[k] + margin) { got = k; break; }
        if (got >= 0) { G->mode[i] = CHASE; G->mtx[i] = ptx[got]; G->mty[i] = pty[got]; }
        else if (G->mode[i] == CHASE) G->mode[i] = rest_mode(G, i);
    }
}

static int catch_disc(Game *G, int i) {
    const UsConfig *c = G->c;
    const Flight *f = &G->f;
    double x = G->px[i], y = G->py[i];
    G->res.completions++;
    G->res.hcomp += f->huck;
    G->res.bcomp += f->is_break;
    if (!(x >= 0.0 && x <= c->L && y >= 0.0 && y <= c->W)) { end_possession(G, US_OUT_OF_BOUNDS, x, y); return 1; }
    if (x >= c->L - c->ez) { end_possession(G, US_SCORE, x, y); return 1; }
    G->in_flight = 0;
    G->holder = i;
    if (x < c->ez) { x = c->ez; G->px[i] = x; }
    G->dx = x; G->dy = y;
    G->s_open = compute_s_open(G, y);
    G->stall = 0.0;
    G->mode[i] = HOLD;
    G->ready_time = G->t + G->P[i].release_time;
    G->next_decision = G->ready_time;
    G->reset_player = -1;
    for (int j = O0; j < O0 + 7; j++) if (j != i && G->mode[j] == CHASE) G->mode[j] = CLEAR;
    for (int j = D0; j < D0 + 7; j++) if (G->mode[j] == CHASE) G->mode[j] = rest_mode(G, j);
    int others[NSLOT], n = 0;
    for (int j = O0; j < O0 + 7; j++) if (j != i) others[n++] = j;
    assign_slots(G, others, x, y);
    if (G->keff >= 1) {   /* zone and flex re-form around the new thrower ... */
        const UsDefense *gd = DG;
        if (gd->transition_throws < 9.5 && G->res.completions >= (int)floor(gd->transition_throws + 0.5))
            G->keff = 0;      /* ... or, after N completions, switch to person D (zone-to-person flex) */
        assign_defense(G, 1);
        for (int j = D0; j < D0 + 7; j++) if (G->mode[j] == ZONE || G->mode[j] == DEF) G->mode[j] = rest_mode(G, j);
    }
    return 0;
}

static int try_catch(Game *G, int pl, int contested, int near_def, double vdisc, const double *scale) {
    const UsConfig *c = G->c;
    double pc = G->P[pl].hands - c->speed_drop_per_ms * maxd(vdisc - 12.0, 0.0);
    if (contested || near_def) pc -= c->contest_drop;
    if (G->f.huck) pc -= c->huck_drop;  /* over-the-shoulder catch at full sprint */
    if (G->f.ttype == HAM) pc -= c->hammer_drop;  /* upside-down disc dropping in */
    return rng_u(&G->rng) < pc * scale[pl];
}

static int block_disc(Game *G, int contested, double curx, double cury) {
    const Flight *f = &G->f;
    int kind = (contested || dist2(G->px[f->receiver], G->py[f->receiver], curx, cury) < 3.0) ? US_BLOCK : US_INTERCEPTION;
    end_possession(G, kind, curx, cury);
    return 1;
}

static int resolve_touch(Game *G, const int *touch, const double *dmin, double *scale, double curx, double cury) {
    const UsConfig *c = G->c;
    Flight *f = &G->f;
    double since = G->t - f->t0;
    int anyO = 0, anyD = 0;
    for (int i = 0; i < NP; i++) {
        if (!touch[i]) continue;
        if (i >= O0 && i < O0 + 7) anyO = 1;
        else {
            anyD = 1;
            double huck_dl = f->huck ? c->huck_read_extra : 0.0;
            if (since < G->P[i].reaction_time + c->disc_read_delay + huck_dl + help_delay(G, i, f->receiver))
                scale[i] *= c->reflex_block;
        }
    }
    int near_def = 0;
    for (int j = D0; j < D0 + 7 && !near_def; j++) near_def = dist2(G->px[j], G->py[j], curx, cury) < 2.0;
    double vdisc = f->dist / f->T;
    /* best toucher of each side */
    int o = -1, d = -1;
    double bo = DBL_MAX, bd = DBL_MAX;
    for (int i = O0; i < O0 + 7; i++) if (touch[i]) { double k = dmin[i] - 0.5 * (i == f->receiver); if (k < bo) { bo = k; o = i; } }
    for (int i = D0; i < D0 + 7; i++) if (touch[i]) { double k = dmin[i] - 0.5 * (i == f->receiver); if (k < bd) { bd = k; d = i; } }
    if (anyD && anyO) {
        double logit = c->contest_base + c->contest_height_weight * (G->effh[o] - G->effh[d]) +
                       c->contest_position_weight * (dmin[o] < dmin[d] ? 1.0 : -1.0);
        if (rng_u(&G->rng) < sigm(logit)) {
            if (try_catch(G, o, 1, near_def, vdisc, scale)) return catch_disc(G, o);
            end_possession(G, US_DROP, curx, cury);
            return 1;
        }
        if (rng_u(&G->rng) < c->block_prob * scale[d]) return block_disc(G, 1, curx, cury);
        if (try_catch(G, o, 1, near_def, vdisc, scale)) return catch_disc(G, o);
        end_possession(G, US_DROP, curx, cury);
        return 1;
    }
    if (anyD) {
        if (rng_u(&G->rng) < c->block_prob * scale[d]) return block_disc(G, 0, curx, cury);
        f->touched |= 1 << d;
        return 0;
    }
    if (try_catch(G, o, 0, near_def, vdisc, scale)) return catch_disc(G, o);
    end_possession(G, US_DROP, curx, cury);
    return 1;
}

static int advance_flight(Game *G) {
    const UsConfig *c = G->c;
    Flight *f = &G->f;
    double s = (G->t - f->t0) / f->T, s_c = mind(s, f->s_end);
    double curx, cury;
    flight_pos(G, s_c, &curx, &cury);
    double prevx = f->prevx, prevy = f->prevy;
    f->prevx = curx; f->prevy = cury;
    double h_disc = flight_height(c, f->peak, s_c, f->T, f->h0);
    int catchable = s_c * f->dist >= c->double_team_radius;
    if (catchable) {
        int touch[NP], any = 0, hb = hist_idx(G, 0);
        double dmin[NP], u[NP], scale[NP];
        for (int i = 0; i < NP; i++) {
            double P0x = G->hlen ? G->hpx[hb][i] : G->px[i], P0y = G->hlen ? G->hpy[hb][i] : G->py[i];
            double r0x = prevx - P0x, r0y = prevy - P0y, r1x = curx - G->px[i], r1y = cury - G->py[i];
            double drx = r1x - r0x, dry = r1y - r0y, dl2 = drx * drx + dry * dry;
            u[i] = dl2 > 1e-9 ? clampd(-(r0x * drx + r0y * dry) / maxd(dl2, 1e-9), 0.0, 1.0) : 1.0;
            double rx = r0x + u[i] * drx, ry = r0y + u[i] * dry;
            dmin[i] = sqrt(rx * rx + ry * ry);
            int active = G->t >= G->ground_until[i];
            double radius = G->reach[i] * (h_disc > G->standing[i] ? c->jump_reach_mult : 1.0);   /* jumping */
            touch[i] = active && h_disc <= G->maxreach[i] && dmin[i] <= radius && !((f->touched >> i) & 1);
            scale[i] = h_disc < c->low_disc ? c->low_mult : 1.0;
        }
        double ext = c->layout_extension;
        for (int i = 0; i < NP; i++) {
            int active = G->t >= G->ground_until[i];
            int passing = u[i] < 1.0 || s >= f->s_end;
            if (!(active && !touch[i] && G->mode[i] == CHASE && dmin[i] <= G->reach[i] + ext && passing
                  && h_disc <= G->standing[i])) continue;   /* no layouts for discs overhead */
            if ((f->bid_done >> i) & 1) continue;
            f->bid_done |= 1 << i;
            double aggr = (i / 7) == G->offense ? G->og[i / 7]->bid_aggr : G->dg[i / 7]->bid_aggr;
            if (rng_u(&G->rng) < aggr) {
                touch[i] = 1;
                scale[i] *= c->layout_success_scale * (1.0 - 0.5 * (dmin[i] - G->reach[i]) / ext);
                G->ground_until[i] = G->t + c->layout_ground_time;
                G->mode[i] = GROUND;
            }
        }
        for (int i = 0; i < NP; i++) any |= touch[i];
        if (any) return resolve_touch(G, touch, dmin, scale, curx, cury);
    }
    if (s >= f->s_end) {
        int out = !(curx >= 0.0 && curx <= c->L && cury >= 0.0 && cury <= c->W);
        end_possession(G, out ? US_OUT_OF_BOUNDS : US_THROWAWAY, curx, cury);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ main loop */
static void tick(Game *G) {
    const UsConfig *c = G->c;
    if (G->holder >= 0 && G->t >= G->ready_time && G->t >= G->next_decision && G->t >= G->fake_until) {
        G->next_decision = G->t + c->decision_interval;
        thrower_decision(G);
        if (G->res.outcome >= 0) return;
    }
    double tx[NP], ty[NP], spd[NP], tvx[NP], tvy[NP];
    for (int i = 0; i < NP; i++) { tx[i] = G->px[i]; ty[i] = G->py[i]; spd[i] = 0.0; tvx[i] = tvy[i] = 0.0; }
    if (G->in_flight) update_chasers(G);
    maybe_switch(G);
    offense_targets(G, tx, ty, spd);
    defense_targets(G, tx, ty, spd, tvx, tvy);
    hist_push(G);
    const double slack = c->boundary_slack;
    for (int i = 0; i < NP; i++) {
        int frozen = G->t < G->ground_until[i];
        step_one(&G->px[i], &G->py[i], &G->vx[i], &G->vy[i], tx[i], ty[i], spd[i], G->P[i].accel, G->brake[i],
                 TOP(i), tvx[i], tvy[i], c->dt, frozen);
        G->px[i] = clampd(G->px[i], -slack, c->L + slack);
        G->py[i] = clampd(G->py[i], -slack, c->W + slack);
    }
    if (G->holder >= 0) { G->dx = G->px[G->holder]; G->dy = G->py[G->holder]; }
    G->t += c->dt;
    if (G->in_flight && advance_flight(G)) return;
    if (G->holder >= 0) {
        if (G->t >= G->grace_until && marker_set(G)) G->stall += c->stall_rate * c->dt;
        if (G->stall > G->res.max_stall) G->res.max_stall = G->stall;
        if (G->stall >= c->stall_max) end_possession(G, US_STALL, G->dx, G->dy);
    }
    if (G->t - G->poss_t0 > c->max_poss_time && G->res.outcome < 0)
        end_possession(G, US_TIMEOUT, G->in_flight ? G->f.p1x : G->dx, G->in_flight ? G->f.p1y : G->dy);
}

static void run_possession(Game *G, double dx, double dy, int fresh) {
    if (dy < 0) {
        double m = G->c->start_y_margin;
        dx = G->c->start_x;
        dy = m + rng_u(&G->rng) * (G->c->W - 2 * m);
    }
    start_possession(G, dx, dy, fresh);
    while (G->res.outcome < 0) tick(G);
}

static void flip(Game *G) {
    const double L = G->c->L, W = G->c->W;
    for (int i = 0; i < NP; i++) {
        G->px[i] = L - G->px[i]; G->py[i] = W - G->py[i]; G->vx[i] = -G->vx[i]; G->vy[i] = -G->vy[i];
        for (int k = 0; k < G->hmax; k++) {
            G->hpx[k][i] = L - G->hpx[k][i]; G->hpy[k][i] = W - G->hpy[k][i];
            G->hvx[k][i] = -G->hvx[k][i]; G->hvy[k][i] = -G->hvy[k][i];
        }
    }
    G->offense = 1 - G->offense;
}

static void accumulate(UsResult *o, const Poss *p) {
    o->throws += p->throws; o->completions += p->completions; o->hucks += p->hucks; o->huck_completions += p->hcomp;
    o->breaks += p->breaks; o->break_completions += p->bcomp; o->hammers += p->hammers; o->fakes += p->fakes;
    o->duration += p->duration;
    if (p->max_stall > o->max_stall) o->max_stall = p->max_stall;
    o->sum_stall_at_throw += p->sum_stall; o->sum_p_chosen += p->sum_p;
    int32_t *counts = &o->n_score;
    counts[p->outcome]++;
    o->n_poss++;
    o->outcome = p->outcome; o->offense = p->offense; o->end_x = p->end_x; o->end_y = p->end_y;
}

static void init_game(Game *G, const UsConfig *c, const UsOffense *offs, const UsDefense *defs, const UsGameSpec *sp) {
    memset(G, 0, sizeof *G);
    G->c = c;
    G->P = sp->players;
    for (int t = 0; t < 2; t++) {
        G->og[t] = sp->off_idx[t] >= 0 ? &offs[sp->off_idx[t]] : &offs[0];
        G->dg[t] = sp->def_idx[t] >= 0 ? &defs[sp->def_idx[t]] : &defs[0];
        double fz = clampd(G->dg[t]->force, 0.0, 4.999999999);
        G->force[t] = (int)fz;
        G->nzone[t] = (int)floor(clampd(G->dg[t]->zone, 0.0, 1.0) * 7.0 + 0.5);
    }
    rng_seed(&G->rng, sp->seed);
    for (int i = 0; i < NP; i++) {
        const UsPlayer *p = &sp->players[i];
        G->brake[i] = p->accel * (c->brake_base + c->brake_agility * p->agility);
        G->reach[i] = c->reach_base + c->reach_height * (p->height - 1.6) + c->reach_vert * p->vertical;
        G->effh[i] = p->height + 0.6 * p->vertical;
        G->standing[i] = p->height * c->standing_reach_ratio;   /* reach envelope */
        G->maxreach[i] = G->standing[i] + p->vertical;
        G->mode[i] = SET; G->slot[i] = G->zslot[i] = G->match[i] = G->def_of[i] = -1;
        G->ground_until[i] = -1.0;
    }
    G->hmax = (int)ceil(1.0 / c->dt) + 2;   /* long enough for reaction + blind delay */
    if (G->hmax > HCAP) G->hmax = HCAP;
    G->last_fake = -10.0;
    G->s_open = 1.0;
    G->holder = -1;
    for (int q = 0; q < US_N_MODEL_W; q++) G->model_w[q] = (&c->model_w0)[q];
}

static void run_spec(const UsConfig *c, const UsOffense *offs, const UsDefense *defs, const UsGameSpec *sp, UsResult *out) {
    Game G;
    init_game(&G, c, offs, defs, sp);
    memset(out, 0, sizeof *out);
    out->scorer = -1;
    run_possession(&G, sp->start_x, sp->start_y, 1);
    accumulate(out, &G.res);
    if (sp->mode == 1) {
        while (G.res.outcome != US_SCORE && out->n_poss < (int)c->max_point_poss) {
            double ex = G.res.end_x, ey = G.res.end_y;
            flip(&G);
            double nx = clampd(c->L - ex, c->ez, c->L - c->ez - 0.01), ny = clampd(c->W - ey, 0.5, c->W - 0.5);
            run_possession(&G, nx, ny, 0);
            accumulate(out, &G.res);
        }
    }
    if (G.res.outcome == US_SCORE) out->scorer = G.res.offense;
}

int32_t us_sim_batch(const UsConfig *cfg, const UsOffense *offs, int32_t n_off, const UsDefense *defs, int32_t n_def,
                     const UsGameSpec *specs, int32_t n_specs, UsResult *out, int32_t n_threads) {
    if (!cfg || !offs || !defs || !specs || !out || n_off <= 0 || n_def <= 0) return -1;
    for (int i = 0; i < n_specs; i++)
        for (int t = 0; t < 2; t++)
            if (specs[i].off_idx[t] >= n_off || specs[i].def_idx[t] >= n_def) return -2;
#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (int i = 0; i < n_specs; i++) run_spec(cfg, offs, defs, &specs[i], &out[i]);
    (void)n_threads;
    return 0;
}
