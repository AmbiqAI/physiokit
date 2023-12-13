/**
 * @file pk_hrv.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Heart Rate Variability
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_HRV_H
#define __PK_HRV_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

typedef struct {
    // Deviation-based
    float32_t meanNN;
    float32_t sdNN;
    // Difference-based
    float32_t rmsSD;
    float32_t sdSD;
    // Normalized
    float32_t cvNN;
    float32_t cvSD;
    // Robust
    float32_t medianNN;
    float32_t madNN;
    float32_t mcvNN;
    float32_t iqrNN;
    float32_t prc20NN;
    float32_t prc80NN;

    // Extrema
    uint32_t nn50;
    uint32_t nn20;
    float32_t pnn50;
    float32_t pnn20;
    float32_t minNN;
    float32_t maxNN;
} hrv_td_metrics_t;

typedef struct {
} hrv_fd_metrics_t;

uint32_t
pk_hrv_compute_time_metrics_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_td_metrics_t *metrics);

uint32_t
pk_hrv_compute_freq_metrics_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_fd_metrics_t *metrics);

#ifdef __cplusplus
}
#endif

#endif // __PK_HRV_H
