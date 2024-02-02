/**
 * @file pk_transform.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Transforms
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_TRANSFORM_H
#define __PK_TRANSFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

uint32_t
rescale_signal_f32(float32_t *pSrc, float32_t oldMin, float32_t oldMax, float32_t newMin, float32_t newMax, uint32_t blockSize, uint8_t clip, float32_t *pRst);

#ifdef __cplusplus
}
#endif

#endif // __PK_TRANSFORM_H
