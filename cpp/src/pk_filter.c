/**
 * @file pk_filter.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Filtering
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


uint32_t
pk_resample_signal_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample)
{
    /**
     * @brief Resample signal by upsampling followed by downsamping
     *
     */
    return 1;
}

uint32_t
pk_resample_categorical_u32(uint32_t *pSrc, uint32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample)
{
    return 1;
}

uint32_t
pk_linear_downsample_f32(float32_t *pSrc, uint32_t srcSize, uint32_t srcFs, float32_t *pRst, uint32_t rstSize, uint32_t rstFs)
{
    /**
     * @brief Basic downsampling using linear interpolation
     *
     */
    float32_t xi, yl, yr;
    uint32_t xl, xr;
    float32_t ratio = ((float32_t)srcFs) / rstFs;
    for (size_t i = 0; i < rstSize; i++)
    {
        xi = i * ratio;
        xl = floorf(xi);
        xr = ceilf(xi);
        yl = pSrc[xl];
        yr = pSrc[xr];
        pRst[i] = xl == xr ? yl : yl + (xi - xl) * ((yr - yl) / (xr - xl));
    }
    return 0;
}

uint32_t
pk_blackman_coefs_f32(float32_t *coefs, uint32_t len) {
    /**
     * @brief Generate Blackman window coefficients
     *
     */
    for (size_t i = 0; i < len; i++)
    {
        int32_t n = 2*i - len + 1;
        coefs[i] = 0.42 + 0.5*cosf(PI*n/(len-1)) + 0.08*cosf(2*PI*n/(len-1));
    }
    return 0;
}

uint32_t
pk_init_biquad_filter_f32(biquad_filt_f32_t *biquadCtx)
{
    /**
     * @brief Initialize biquad filter
     *
     */
    arm_biquad_cascade_df1_init_f32(biquadCtx->inst, biquadCtx->numSecs, biquadCtx->sos, biquadCtx->state);
    return 0;
}

uint32_t
pk_apply_biquad_filter_f32(biquad_filt_f32_t *biquadCtx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize)
{
    /**
     * @brief Apply biquad filter to signal
     */
    arm_biquad_cascade_df1_f32(biquadCtx->inst, pSrc, pResult, blockSize);
    return 0;
}

uint32_t
pk_apply_biquad_filtfilt_f32(biquad_filt_f32_t *biquadCtx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *state)
{
    /**
     * @brief Apply biquad filter forward-backward to signal
     */
    // Forward pass
    arm_fill_f32(0, biquadCtx->inst->pState, 4*biquadCtx->numSecs);
    arm_biquad_cascade_df1_f32(biquadCtx->inst, pSrc, pResult, blockSize);
    for (size_t i = 0; i < blockSize; i++){ state[i] = pResult[blockSize-1-i]; }
    // Backward pass
    arm_fill_f32(0, biquadCtx->inst->pState, 4*biquadCtx->numSecs);
    arm_biquad_cascade_df1_f32(biquadCtx->inst, state, pResult, blockSize);
    for (size_t i = 0; i < blockSize; i++){ state[i] = pResult[blockSize-1-i]; }
    for (size_t i = 0; i < blockSize; i++){ pResult[i] = state[i]; }
    return 0;
}

uint32_t
pk_quotient_filter_mask_u32(uint32_t *data, uint8_t *mask, uint32_t dataLen, uint32_t iterations, float32_t lowcut, float32_t highcut)
{
    /**
     * @brief Apply quotient filter mask to signal
     */
    int32_t m = -1, n = -1;
    uint32_t numFound;
    float32_t q;
    for (size_t iter = 0; iter < iterations; iter++)
    {
        numFound = 0;
        for (size_t i = 0; i < dataLen; i++)
        {
            if (mask[i] == 0)
            {
                // Find first value
                if (m == -1)
                {
                    m = i;
                    n = -1;
                // Find second value
                }
                else if (n == -1)
                {
                    n = i;
                }
                // Compute quotient and check if in range
                if (m != -1 && n != -1)
                {
                    q = (float32_t)data[m] / (float32_t)data[n];
                    if (q < lowcut || q > highcut)
                    {
                        mask[m] = 1;
                        numFound++;
                    }
                    m = -1;
                    n = -1;
                }
            }
        }
        // Stop early if no new values found
        if (numFound == 0)
        {
            break;
        }
    }

    return 0;
}


uint32_t
pk_smooth_signal_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *wBuffer, uint32_t windowSize)
{
    /**
     * @brief Smooth signal using moving average filter
     */

    // Utilize dot product to compute moving average
    uint32_t halfWindowSize = windowSize / 2;
    arm_fill_f32(1.0f / windowSize, wBuffer, windowSize);
    for (size_t i = 0; i < blockSize - windowSize; i++)
    {
        arm_dot_prod_f32(pSrc + i, wBuffer, windowSize, pResult + i + halfWindowSize);
    }
    // Replicate first and last values at the edges
    arm_fill_f32(pResult[halfWindowSize], pResult, halfWindowSize);
    uint32_t dpEnd = blockSize - windowSize - 1 + halfWindowSize;
    arm_fill_f32(pResult[dpEnd], &pResult[dpEnd], blockSize - dpEnd);

    return 0;
}


uint32_t
pk_standardize_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t epsilon)
{
    /**
     * @brief Standardize signal: y = (x - mu) / std. Provides safegaurd against small st devs
     *
     */
    float32_t mu, std;
    pk_mean_f32(pSrc, &mu, blockSize);
    pk_std_f32(pSrc, &std, blockSize);
    std = std + epsilon;
    arm_offset_f32(pSrc, -mu, pResult, blockSize);
    arm_scale_f32(pResult, 1.0f / std, pResult, blockSize);
    return 0;
}
