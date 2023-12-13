/**
 * @file pk_imu.c
 * @author Adam Page (adam.page@ambiq.com)
 * @brief PhysioKit: IMU
 * @version 1.0
 * @date 2023-12-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "arm_math.h"

#include "pk_imu.h"

uint32_t
pk_imu_compute_enmo_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *enmo, uint32_t blockSize){
    /**
     * @brief Compute ENMO from accelerometer data
     *
     */

    // enmo = np.maximum(np.sqrt(x**2 + y**2 + z**2) - 1, 0)
    for (size_t i = 0; i < blockSize; i++)
    {
        enmo[i] = sqrtf(powf(x[i], 2) + powf(y[i], 2) + powf(z[i], 2) - 1.0f);
    }
    return 0;
}

uint32_t
pk_imu_compute_tilt_angles_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *xAngle, float32_t* yAngle, float32_t *zAngle, uint32_t blockSize){
    /**
     * @brief Compute tilt angles from accelerometer data
     *
     */

    // xAngle = np.arctan2(x, np.sqrt(y**2 + z**2))
    // yAngle = np.arctan2(y, np.sqrt(x**2 + z**2))
    // zAngle = np.arctan2(z, np.sqrt(x**2 + y**2))
    for (size_t i = 0; i < blockSize; i++)
    {
        xAngle[i] = atan2f(x[i], sqrtf(powf(y[i], 2) + powf(z[i], 2)));
        yAngle[i] = atan2f(y[i], sqrtf(powf(x[i], 2) + powf(z[i], 2)));
        zAngle[i] = atan2f(z[i], sqrtf(powf(x[i], 2) + powf(y[i], 2)));
    }
    return 0;
}


uint32_t
pk_imu_compute_pitch_roll_f32(float32_t *x, float32_t *y, float32_t *z, float32_t *pitch, float32_t *roll, uint32_t blockSize){
    /**
     * @brief Compute pitch, and roll from accelerometer data
     *
     */

    // pitch = np.arctan2(-x, np.sqrt(y**2 + z**2))
    // roll = np.arctan2(y, z)
    for (size_t i = 0; i < blockSize; i++)
    {
        pitch[i] = atan2f(-x[i], sqrtf(powf(y[i], 2) + powf(z[i], 2)));
        roll[i] = atan2f(y[i], z[i]);
    }
    return 0;
}
