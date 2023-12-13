/**
 * @file pk_rsp.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: RSP
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_RSP_H
#define __PK_RSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

typedef struct
{
    float32_t peakWin; // 0.5
    float32_t breathWin; // 2.0
    float32_t breathOffset; // 0.05
    float32_t peakDelayWin; // 0.3
    uint32_t sampleRate;
    // State requires 4*ppgLen
    float32_t *state;
    uint32_t *peaks;
} rsp_peak_f32_t;

uint32_t
pk_rsp_find_peaks_f32(rsp_peak_f32_t* ctx, float32_t *rsp, uint32_t rspLen, uint32_t *peaks);

uint32_t
pk_rsp_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals);

uint32_t
pk_rsp_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR, float32_t minDelta);

float32_t
pk_rsp_compute_respiratory_rate_from_rr_intervals(uint32_t *rrIntervals, uint32_t *mask, uint32_t numPeaks, uint32_t sampleRate);

#ifdef __cplusplus
}
#endif

#endif // __PK_RSP_H
