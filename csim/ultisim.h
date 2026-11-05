/* ultisim C core: public API. Struct layouts live in the generated us_structs.h
 * (from ultisim/cabi.py) so Python and C always agree. */
#ifndef ULTISIM_H
#define ULTISIM_H

#include <stdint.h>
#include "us_structs.h"

#ifdef _WIN32
#define US_API __declspec(dllexport)
#else
#define US_API __attribute__((visibility("default")))
#endif

US_API int32_t us_abi_version(void);
US_API int32_t us_max_threads(void);
US_API int32_t us_has_openmp(void);
US_API double us_smoke(int64_t n);
US_API int64_t us_sizeof_config(void);
US_API int64_t us_sizeof_spec(void);
US_API int64_t us_sizeof_result(void);
US_API int64_t us_sizeof_offense(void);
US_API int64_t us_sizeof_defense(void);

/* Simulate n_specs independent games (possessions or full points) on up to n_threads threads.
 * Returns 0 on success. Each spec picks genomes by index into offs / defs. */
US_API int32_t us_sim_batch(const UsConfig *cfg, const UsOffense *offs, int32_t n_off,
                            const UsDefense *defs, int32_t n_def, const UsGameSpec *specs,
                            int32_t n_specs, UsResult *out, int32_t n_threads);

/* Deterministic building blocks, exported for exact parity tests against the Python reference. */
US_API double us_time_to_reach(double px, double py, double vx, double vy, double top, double accel,
                               double brake, double reach, double qx, double qy, double delay, double extra);
US_API void us_flight_point(double p0x, double p0y, double cx, double cy, double p1x, double p1y,
                            double T, double s, double tail, double *out2);
US_API double us_error_sigma(const UsConfig *c, double skill, int32_t ttype, double dist,
                             int32_t is_break, int32_t marked, int32_t curved);
US_API double us_mark_block_prob(const UsConfig *c, double hx, double hy, double ex, double ey,
                                 int32_t ttype, double center, double width, double marking,
                                 double break_skill, int32_t fake_active);
US_API void us_step(int32_t n, double *px, double *py, double *vx, double *vy, const double *tx,
                    const double *ty, const double *speed, const double *accel, const double *brake,
                    const double *top, const double *tvel_xy, double dt, const int32_t *frozen);
US_API void us_hungarian(int32_t n, int32_t m, const double *cost, int32_t *row_to_col);
US_API double us_flight_height(const UsConfig *c, double peak, double s, double T, double h0);

#endif
