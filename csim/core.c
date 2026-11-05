#include "ultisim.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int32_t us_abi_version(void) { return ULTISIM_ABI_VERSION; }

int32_t us_max_threads(void) {
#ifdef _OPENMP
    return (int32_t)omp_get_max_threads();
#else
    return 1;
#endif
}

double us_smoke(int64_t n) {
    double s = 0.0;
    for (int64_t i = 1; i <= n; i++) s += 1.0 / ((double)i * (double)i);
    return s;
}

int32_t us_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int64_t us_sizeof_config(void) { return (int64_t)sizeof(UsConfig); }
int64_t us_sizeof_spec(void) { return (int64_t)sizeof(UsGameSpec); }
int64_t us_sizeof_result(void) { return (int64_t)sizeof(UsResult); }
int64_t us_sizeof_offense(void) { return (int64_t)sizeof(UsOffense); }
int64_t us_sizeof_defense(void) { return (int64_t)sizeof(UsDefense); }
