/**
 * @file pk_filter.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Filtering
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_FILTER_H
#define __PK_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

typedef struct
{
    arm_biquad_casd_df1_inst_f32 *inst;
    uint8_t numSecs;
    const float32_t *sos;
    float32_t *state;
} biquad_filt_f32_t;

uint32_t
pk_resample_signal_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample);

uint32_t
pk_resample_categorical_u32(uint32_t *pSrc, uint32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample);

uint32_t
pk_linear_downsample_f32(float32_t *pSrc, uint32_t srcSize, uint32_t srcFs, float32_t *pRst, uint32_t rstSize, uint32_t rstFs);

uint32_t
pk_init_biquad_filter_f32(biquad_filt_f32_t *biquadCtx);

uint32_t
pk_apply_biquad_filter_f32(biquad_filt_f32_t *biquadCtx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

uint32_t
pk_apply_biquad_filtfilt_f32(biquad_filt_f32_t *biquadCtx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *state);

uint32_t
pk_quotient_filter_mask_u32(uint32_t *data, uint8_t *mask, uint32_t dataLen, uint32_t iterations, float32_t lowcut, float32_t highcut);

uint32_t
pk_smooth_signal_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *wBuffer, uint32_t windowSize);

uint32_t
pk_standardize_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t epsilon);

#ifdef __cplusplus
}
#endif

#endif // __PK_FILTER_H
