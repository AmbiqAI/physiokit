/**
 * @file pk_imu.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: IMU
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PK_IMU_H
#define __PK_IMU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "arm_math.h"

uint32_t
pk_imu_compute_enmo_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *enmo, uint32_t blockSize);

uint32_t
pk_imu_compute_tilt_angles_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *xAngle, float32_t* yAngle, float32_t *zAngle, uint32_t blockSize);

uint32_t
pk_imu_compute_pitch_roll_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *pitch, float32_t *roll, uint32_t blockSize);

#ifdef __cplusplus
}
#endif

#endif // __PK_IMU_H
