# IMU Processing

Inertial measurement units (IMUs) are electronic devices that measure and report a body's specific force (acceleration and angular velocity) and orientation (roll, pitch, and yaw). IMU signals are often used to measure activity, posture, and gait. In PhysioKit, we provide a variety of routines for processing IMU signals.

## Compute ENMO

Euclidean Norm Minus One (ENMO) is a measure of activity intensity. ENMO is calculated as the square root of the sum of the squared acceleration values minus one. ENMO is often used to measure activity intensity from IMU signals.

## Compute Z-Angle

Z-Angle is a measure of posture. Z-Angle is calculated as the arctangent of the ratio of the vertical acceleration to the horizontal acceleration. Z-Angle is often used to measure posture from IMU signals. Z-Angle is also extremely useful for detecting both falls and sleep.

## Compute "Counts"

"Counts" is a measure of activity intensity and typically reported by actigraphy watches. Unfortunately, "Counts" is not a standardized measure and is calculated differently by different manufacturers. In PhysioKit, we compute counts based on algorithms reported by [ActiGraph](https://doi.org/10.1038/s41598-022-16003-x).

---

## API

[Refer to IMU API for more details](../api/imu.md)
