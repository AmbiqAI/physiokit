/**
 * @file pk_ecg.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: ECG
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
#include "pk_ecg.h"

uint32_t
pk_ecg_square_filter_mask(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR) {
    /**
     * @brief Filter RR intervals
     */
    float32_t lowcut = minRR*sampleRate;
    float32_t highcut = maxRR*sampleRate;
    for (size_t i = 0; i < numPeaks; i++)
    {
        mask[i] = (rrIntervals[i] < lowcut) || (rrIntervals[i] > highcut) ? 1 : 0;
    }
    return 0;
}

uint32_t
pk_ecg_find_peaks(ecg_peak_f32_t *ctx, float32_t *ecg, uint32_t ecgLen, uint32_t *peaks)
{
    /**
     * @brief Find R peaks in PPG signal
     */

    uint32_t qrsGradLen = (uint32_t)(ctx->sampleRate * ctx->qrsWin + 1);
    uint32_t avgGradLen = (uint32_t)(ctx->sampleRate * ctx->avgWin + 1);

    uint32_t minQrsDelay = (uint32_t)(ctx->sampleRate * ctx->qrsDelayWin + 1);
    uint32_t minQrsWidth = 0;

    float32_t *absGrad = &ctx->state[0 * ecgLen];
    float32_t *qrsGrad = &ctx->state[1 * ecgLen];
    float32_t *avgGrad = &ctx->state[2 * ecgLen];
    float32_t *wBuffer = &ctx->state[3 * ecgLen];

    // Compute absolute gradient
    pk_gradient_f32(ecg, absGrad, ecgLen);
    arm_abs_f32(absGrad, absGrad, ecgLen);

    // Smooth gradients
    pk_smooth_signal_f32(absGrad, qrsGrad, ecgLen, wBuffer, qrsGradLen);
    pk_smooth_signal_f32(qrsGrad, avgGrad, ecgLen, wBuffer, avgGradLen);

    // Min QRS height
    arm_scale_f32(avgGrad, ctx->qrsPromWeight, avgGrad, ecgLen);

    // Subtract average gradient as threshold
    arm_sub_f32(qrsGrad, avgGrad, qrsGrad, ecgLen);

    uint32_t riseEdge, fallEdge, peakDelay, peakLen, peak;
    float32_t peakVal;
    uint32_t numPeaks = 0;
    int32_t m = -1, n = -1;
    for (size_t i = 1; i < ecgLen; i++)
    {
        riseEdge = qrsGrad[i - 1] <= 0 && qrsGrad[i] > 0;
        fallEdge = qrsGrad[i - 1] > 0 && qrsGrad[i] <= 0;
        if (riseEdge)
        {
            m = i;
        }
        else if (fallEdge && m != -1)
        {
            n = i;
        }
        // If detected
        if (m != -1 && n != -1)
        {
            peakLen = n - m + 1;
            arm_max_f32(&ecg[m], peakLen, &peakVal, &peak);
            peak += m;
            peakDelay = numPeaks > 0 ? peak - peaks[numPeaks - 1]  : minQrsDelay;

            if (peakLen >= minQrsWidth && peakDelay >= minQrsDelay)
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
pk_ecg_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals) {
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
pk_ecg_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate, float32_t minRR, float32_t maxRR, float32_t minDelta) {
    /**
     * @brief Filter RR intervals
     */
    // Filter rri w/ square filter
    pk_ecg_square_filter_mask(rrIntervals, numPeaks, mask, sampleRate, minRR, maxRR);
    // Filter rri w/ quotient filter
    pk_quotient_filter_mask_u32(rrIntervals, mask, numPeaks, 2, 1 - minDelta, 1 + minDelta);
    return 0;
}

float32_t
pk_ecg_compute_heart_rate_from_rr_intervals(uint32_t *rrIntervals, uint32_t *mask, uint32_t numPeaks, uint32_t sampleRate){
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

uint32_t
pk_ecg_derive_respiratory_rate(uint32_t *peaks, uint32_t *rrIntervals, uint32_t numPeaks, uint32_t sampleRate, float32_t *respRate) {
    // Will need state to store FFT
    // Interpolate (peaks, rri)
    // Filter to respiratory band
    // Compute FFT and compute power spectrum
    // Find peak in power spectrum
    return 1;
}
