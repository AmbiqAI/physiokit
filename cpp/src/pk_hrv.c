/**
 * @file pk_hrv.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Heart Rate Variability
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <math.h>
#include "arm_math.h"

#include "pk_filter.h"
#include "pk_hrv.h"


uint32_t
pk_hrv_compute_time_metrics_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_td_metrics_t *metrics) {
    // Deviation-based
    metrics->meanNN = 0;
    metrics->sdNN = 0;
    uint32_t numValid = 0;
    for (size_t i = 0; i < numPeaks; i++){
        if (mask[i] == 0) {
            metrics->meanNN += rrIntervals[i];
            metrics->sdNN += rrIntervals[i]*rrIntervals[i];
            numValid++;
        }
    }
    metrics->meanNN /= numValid;
    metrics->sdNN = sqrt(metrics->sdNN/numValid - metrics->meanNN*metrics->meanNN);

    // Difference-based
    float32_t meanSD = 0;
    metrics->rmsSD = 0;
    metrics->sdSD = 0;
    metrics->nn20 = 0;
    metrics->nn50 = 0;
    metrics->minNN = -1;
    metrics->maxNN = -1;
    int32_t v1, v2, v3, v4;
    for (size_t i = 1; i < numPeaks; i++)
    {
        v1 = mask[i - 1] == 0 ? rrIntervals[i - 1] : metrics->meanNN;
        v2 = mask[i] == 0 ? rrIntervals[i] : metrics->meanNN;
        v3 = (v2 - v1);
        v4 = v3*v3;
        meanSD += v3;
        metrics->rmsSD += v4;
        metrics->sdSD += v4;
        if (1000*v4 > 20*20) {
            metrics->nn20++;
        }
        if (1000*v4 > 50*50) {
            metrics->nn50++;
        }
        if (rrIntervals[i] < metrics->minNN || metrics->minNN == -1) {
            metrics->minNN = rrIntervals[i];
        }
        if (rrIntervals[i] > metrics->maxNN || metrics->maxNN == -1) {
            metrics->maxNN = rrIntervals[i];
        }
    }
    meanSD /= (numPeaks-1);
    metrics->rmsSD = sqrt(metrics->rmsSD/(numPeaks-1));
    metrics->sdSD = sqrt(metrics->sdSD/(numPeaks-2) - meanSD*meanSD);
    metrics->pnn20 = 100.0f*metrics->nn20/(numPeaks-1);
    metrics->pnn50 = 100.0f*metrics->nn50/(numPeaks-1);
    // Normalized
    metrics->cvNN = metrics->sdNN/metrics->meanNN;
    metrics->cvSD = metrics->rmsSD/metrics->meanNN;

    // Robust
    metrics->medianNN = 0;
    metrics->madNN = 0;
    metrics->mcvNN = 0;
    // Use mean & std for IQR
    float32_t q1 = metrics->meanNN - 0.6745*metrics->sdNN;
    float32_t q3 = metrics->meanNN + 0.6745*metrics->sdNN;
    metrics->iqrNN = q3 - q1;
    metrics->prc20NN = 0;
    metrics->prc80NN = 0;
    return 0;
}

uint32_t
pk_hrv_compute_freq_metrics_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_fd_metrics_t *metrics){
    return 1;
}
