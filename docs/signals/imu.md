# IMU Processing

Inertial measurement units (IMUs) are electronic devices that measure and report a body's specific force (acceleration and angular velocity) and orientation (roll, pitch, and yaw). IMU signals are often used to measure activity, posture, and gait. In PhysioKit, we provide a variety of routines for processing IMU signals.

## Compute ENMO

Euclidean Norm Minus One (ENMO) is a measure of activity intensity. ENMO is calculated as the square root of the sum of the squared acceleration values minus one. ENMO is often used to measure activity intensity from IMU signals.

???+ example
    In the following example, we load accelerometer data from a user's wrist when performing the following sequence of tasks for 20s each: sitting, walking, and running.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk_imu_accel.html"
    </div>

    From this, we can then easily compute the ENMO:

    ```python
    import physiokit as pk

    # Load accelerometer data
    ax, ay, az = ...

    enmo = pk.imu.compute_enmo(x=ax, y=ay, z=az)
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk_imu_enmo.html"
    </div>

---

## Compute Tilt Angles

3-axis tilt provides insight into a user's orientation. For example, a 3-axis accelerometer placed on a user's wrist can be used to determine the its tilt. The z-angle of the wrist can be extremely useful for both fall and sleep detection applications.

???+ example
    Following the previous example, we compute the tilt angles of the user's wrist.

    ```python
    import physiokit as pk

    # Load accelerometer data
    ax, ay, az = ...

    aax, aay, aaz = pk.imu.compute_tilt_angles(x=ax, y=ay, z=az)
    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk_imu_tilts.html"
    </div>

## Compute "Counts"

"Counts" is a measure of activity intensity and typically reported by actigraphy watches. Unfortunately, "Counts" is not a standardized measure and is calculated differently by different manufacturers. In PhysioKit, we compute counts based on algorithms reported by [ActiGraph](https://doi.org/10.1038/s41598-022-16003-x).

???+ example

    Again, using the 3-axis accelerometer wrist data, we can compute "counts" as follows:


    ```python
    import physiokit as pk

    # Load accelerometer data
    ax, ay, az = ...

    counts = np.sum(pk.imu.compute_counts(
        data=np.vstack((ax, ay, az)).T,
        sample_rate=fs,
        epoch_len=1
    ), axis=1)

    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/pk_imu_counts.html"
    </div>

---

## API

[Refer to IMU API for more details](../api/imu.md)
