/**
 * @file pk_math.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: Math
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_MATH_H
#define __PK_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"

uint32_t
pk_mean_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

uint32_t
pk_std_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

uint32_t
pk_gradient_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

uint32_t
pk_rms_f32(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

#ifdef __cplusplus
}
#endif

#endif // __PK_MATH_H
