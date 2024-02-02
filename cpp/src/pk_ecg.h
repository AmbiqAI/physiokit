/**
 * @file pk_ecg.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: ECG
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __PK_ECG_H
#define __PK_ECG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

typedef struct
{
    float32_t qrsWin; // 0.1
    float32_t avgWin; // 1.0
    float32_t qrsPromWeight; // 1.5
    float32_t qrsMinLenWeight; // 0.4
    float32_t qrsDelayWin; // 0.3
    uint32_t sampleRate;
    // State requires 4*ecgLen
    float32_t *state;
} ecg_peak_f32_t;


uint32_t
pk_ecg_square_filter_mask(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR);

uint32_t
pk_ecg_find_peaks(ecg_peak_f32_t *ctx, float32_t *ecgg, uint32_t ecgLen, uint32_t *peaks);

uint32_t
pk_ecg_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals);

uint32_t
pk_ecg_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR, float32_t minDelta);

float32_t
pk_ecg_compute_heart_rate_from_rr_intervals(uint32_t *rrIntervals, uint32_t *mask, uint32_t numPeaks, uint32_t sampleRate);

uint32_t
pk_ecg_derive_respiratory_rate(uint32_t *peaks, uint32_t *rrIntervals, uint32_t numPeaks, uint32_t sampleRate, float32_t *respRate);

#ifdef __cplusplus
}
#endif

#endif // __PK_ECG_H
