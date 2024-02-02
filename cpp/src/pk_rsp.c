/**
 * @file pk_rsp.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: RSP
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <stdint.h>
#include "arm_math.h"
#include "pk_math.h"
#include "pk_filter.h"
#include "pk_rsp.h"

uint32_t
pk_rsp_find_peaks_f32(rsp_peak_f32_t* ctx, float32_t *rsp, uint32_t rspLen, uint32_t *peaks){
    /**
     * @brief Find inspiratory peaks in RSP signal
     */

    // Apply 1st moving average filter
    float32_t muSqrd;

    uint32_t maPeakLen = (uint32_t)(ctx->sampleRate * ctx->peakWin + 1);
    uint32_t maBeatLen = (uint32_t)(ctx->sampleRate * ctx->breathWin + 1);
    uint32_t minPeakDelay = (uint32_t)(ctx->sampleRate * ctx->peakDelayWin + 1);
    uint32_t minPeakWidth = maPeakLen;

    float32_t *maPeak = &ctx->state[0*rspLen];
    float32_t *maBeat = &ctx->state[1*rspLen];
    float32_t *sqrd = &ctx->state[2*rspLen];
    float32_t *wBuffer = &ctx->state[3*rspLen];

    // Compute squared signal
    for (size_t i = 0; i < rspLen; i++)
    {
        sqrd[i] = rsp[i] > 0 ? rsp[i] * rsp[i] : 0;
    }

    pk_mean_f32(sqrd, &muSqrd, rspLen);
    muSqrd = muSqrd * ctx->breathOffset;

    // Apply peak moving average
    pk_smooth_signal_f32(sqrd, maPeak, rspLen, wBuffer, maPeakLen);

    // Apply beat moving average
    pk_smooth_signal_f32(sqrd, maBeat, rspLen, wBuffer, maBeatLen);
    arm_offset_f32(maBeat, muSqrd, maBeat, rspLen);

    arm_sub_f32(maPeak, maBeat, maPeak, rspLen);

    uint32_t riseEdge, fallEdge, peakDelay, peakLen, peak;
    float32_t peakVal;
    uint32_t numPeaks = 0;
    int32_t m = -1, n = -1;
    for (size_t i = 1; i < rspLen; i++)
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
            arm_max_f32(&rsp[m], peakLen, &peakVal, &peak);
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
pk_rsp_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals) {
    /**
     * @brief Compute respiratory intervals from peaks
     */
    for (size_t i = 1; i < numPeaks; i++)
    {
        rrIntervals[i - 1] = peaks[i] - peaks[i - 1];
    }
    rrIntervals[numPeaks - 1] = rrIntervals[numPeaks - 2];
    return 0;
}


uint32_t
pk_rsp_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR, float32_t minDelta) {
    /**
     * @brief Filter respiratory intervals
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
pk_rsp_compute_respiratory_rate_from_rr_intervals(uint32_t *rrIntervals, uint32_t *mask, uint32_t numPeaks, uint32_t sampleRate) {
    /**
     * @brief Compute respiratory rate from RR intervals
     */
    float32_t respRate = 0;
    uint32_t numValid = 0;
    for (size_t i = 0; i < numPeaks; i++)
    {   if (mask[i] == 0) {
            respRate += (float32_t)sampleRate/rrIntervals[i];
            numValid++;
        }
    }
    respRate /= numValid;
    return respRate;
}
