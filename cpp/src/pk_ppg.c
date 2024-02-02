/**
 * @file pk_ppg.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: PPG
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <math.h>
#include "arm_math.h"

#include "pk_math.h"
#include "pk_filter.h"
#include "pk_ppg.h"

uint32_t
pk_ppg_find_peaks_f32(ppg_peak_f32_t* ctx, float32_t *ppg, uint32_t ppgLen, uint32_t *peaks)
{
    /**
     * @brief Find systolic peaks in PPG signal
     */

    // Apply 1st moving average filter
    float32_t muSqrd;

    uint32_t maPeakLen = (uint32_t)(ctx->sampleRate * ctx->peakWin + 1);
    uint32_t maBeatLen = (uint32_t)(ctx->sampleRate * ctx->beatWin + 1);
    uint32_t minPeakDelay = (uint32_t)(ctx->sampleRate * ctx->peakDelayWin + 1);
    uint32_t minPeakWidth = maPeakLen;

    float32_t *maPeak = &ctx->state[0*ppgLen];
    float32_t *maBeat = &ctx->state[1*ppgLen];
    float32_t *sqrd = &ctx->state[2*ppgLen];
    float32_t *wBuffer = &ctx->state[3*ppgLen];

    // Compute squared signal
    for (size_t i = 0; i < ppgLen; i++)
    {
        sqrd[i] = ppg[i] > 0 ? ppg[i] * ppg[i] : 0;
    }

    pk_mean_f32(sqrd, &muSqrd, ppgLen);
    muSqrd = muSqrd * ctx->beatOffset;

    // Apply peak moving average
    pk_smooth_signal_f32(sqrd, maPeak, ppgLen, wBuffer, maPeakLen);

    // Apply beat moving average
    pk_smooth_signal_f32(sqrd, maBeat, ppgLen, wBuffer, maBeatLen);
    arm_offset_f32(maBeat, muSqrd, maBeat, ppgLen);

    arm_sub_f32(maPeak, maBeat, maPeak, ppgLen);

    uint32_t riseEdge, fallEdge, peakDelay, peakLen, peak;
    float32_t peakVal;
    uint32_t numPeaks = 0;
    int32_t m = -1, n = -1;
    for (size_t i = 1; i < ppgLen; i++)
    {
        riseEdge = maPeak[i - 1] <= 0 && maPeak[i] > 0;
        fallEdge = maPeak[i - 1] > 0 && maPeak[i] <= 0;
        if (riseEdge)
        {
            m = i;
        }
        else if (fallEdge && m != -1)
        {
            n = i;
        }
        // Detected peak
        if (m != -1 && n != -1)
        {
            peakLen = n - m + 1;
            arm_max_f32(&ppg[m], peakLen, &peakVal, &peak);
            peak += m;
            peakDelay = numPeaks > 0 ? peak - peaks[numPeaks - 1] : minPeakDelay;
            if (peakLen >= minPeakWidth && peakDelay >= minPeakDelay)
            {
                peaks[numPeaks++] = peak;
            }
            m = -1;
            n = -1;
        }
    }
    return numPeaks;
}

uint32_t
pk_ppg_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals) {
    /**
     * @brief Compute RR intervals from peaks
     */
    for (size_t i = 1; i < numPeaks; i++)
    {
        rrIntervals[i - 1] = peaks[i] - peaks[i - 1];
    }
    rrIntervals[numPeaks - 1] = rrIntervals[numPeaks - 2];
    return 0;
}

uint32_t
pk_ppg_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR, float32_t minDelta) {
    /**
     * @brief Filter RR intervals
     */
    float32_t lowcut = minRR*sampleRate;
    float32_t highcut = maxRR*sampleRate;

    // Filter out peaks with RR intervals outside of normal range
    uint32_t val;
    uint8_t maskVal;
    for (size_t i = 0; i < numPeaks; i++)
    {
        val = rrIntervals[i];
        maskVal = (val < lowcut) || (val > highcut) ? 1 : 0;
        mask[i] = maskVal;
    }

    pk_quotient_filter_mask_u32(rrIntervals, mask, numPeaks, 2, 1 - minDelta, 1 + minDelta);
    return 0;
}

float32_t
pk_ppg_compute_heart_rate_from_rr_intervals(uint32_t *rrIntervals, uint32_t *mask, uint32_t numPeaks, uint32_t sampleRate){
    /**
     * @brief Compute heart rate from RR intervals
     */
    float32_t heartRate = 0;
    uint32_t numValid = 0;
    for (size_t i = 0; i < numPeaks; i++)
    {   if (mask[i] == 0) {
            heartRate += (float32_t)sampleRate/rrIntervals[i];
            numValid++;
        }
    }
    heartRate /= numValid;
    return heartRate;
}


float32_t
pk_ppg_compute_spo2_from_perfusion_f32(float32_t dc1, float32_t ac1, float32_t dc2, float32_t ac2, float32_t* coefs)
{
    float32_t r = (ac1/dc1)/(ac2/dc2);
    float32_t spo2 = coefs[0]*r*r + coefs[1]*r + coefs[2];
    return spo2;
}

float32_t
pk_ppg_compute_spo2_in_time_f32(float32_t *ppg1, float32_t *ppg2, float32_t ppg1Mean, float32_t ppg2Mean, uint32_t blockSize, float32_t *coefs, float32_t sampleRate)
{
    float32_t ppg1Dc, ppg2Dc, ppg1Ac, ppg2Ac, spo2;

    // Compute DC via mean
    ppg1Dc = ppg1Mean;
    ppg2Dc = ppg2Mean;

    // Assume signals are already filtered

    // Compute AC via RMS
    arm_rms_f32(ppg1, blockSize, &ppg1Ac);
    arm_rms_f32(ppg2, blockSize, &ppg2Ac);

    // Compute SpO2
    spo2 = pk_ppg_compute_spo2_from_perfusion_f32(ppg1Dc, ppg1Ac, ppg2Dc, ppg2Ac, coefs);
    return spo2;
}
