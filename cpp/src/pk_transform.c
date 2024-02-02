/**
 * @file pk_transform.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Transforms
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <stdint.h>
#include "arm_math.h"
#include "pk_transform.h"

uint32_t
rescale_signal_f32(float32_t *pSrc, float32_t oldMin, float32_t oldMax, float32_t newMin, float32_t newMax, uint32_t blockSize, uint8_t clip, float32_t *pRst) {
    /**
     * @brief Rescale signal to new range
     *
     */

    float32_t oldRange = oldMax - oldMin;
    float32_t newRange = newMax - newMin;
    for (size_t i = 0; i < blockSize; i++)
    {
        pRst[i] = (pSrc[i] - oldMin) * newRange / oldRange + newMin;
        if (clip) {
            if (pRst[i] > newMax) {
                pRst[i] = newMax;
            }
            else if (pRst[i] < newMin) {
                pRst[i] = newMin;
            }
        }
    }
    return 0;
}

uint32_t
pk_compute_fft_f32(arm_rfft_instance_f32 *fftCtx, float32_t *pSrc, float32_t *pDst, uint32_t fftLen) {
    arm_rfft_f32(fftCtx, pSrc, pDst);
    return 0;
}
