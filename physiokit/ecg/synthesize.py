import functools
import math

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.ndimage

from .synthetic import EcgPreset, simulate_brisk


def synthesize(
    signal_length: int = 10000,
    sample_rate: float = 1000,
    leads: int = 12,
    heart_rate: float = 60,
    preset: EcgPreset = EcgPreset.SR,
    noise_multiplier: float = 1.0,
    impedance: float = 1.0,
    p_multiplier: float = 1.0,
    t_multiplier: float = 1.0,
    voltage_factor: float = 300,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generate synthetic ECG signal via brisk method.

    Utilize pk.signal.noise methods to make more realistic.

    Leads are indexed as follows:
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    Args:
        signal_length (int, optional): Length of the ECG signal. Defaults to 10000.
        sample_rate (float, optional): ECG sampling frequency. Defaults to 1000.
        leads (int, optional): Number of leads. Defaults to 12.
        heart_rate (float, optional): Mean heart rate. Defaults to 60.
        preset (EcgPreset, optional): ECG preset. Defaults to EcgPreset.SR.
        noise_multiplier (float, optional): Noise multiplier. Defaults to 1.0.
        impedance (float, optional): Impedance. Defaults to 1.0.
        p_multiplier (float, optional): P wave multiplier. Defaults to 1.0.
        t_multiplier (float, optional): T wave multiplier. Defaults to 1.0.
        voltage_factor (float, optional): Voltage factor. Defaults to 300.

    Returns:
        npt.NDArray: Synthetic ECG signals
    """

    _, ecg, segs, fids, _ = simulate_brisk(
        signal_length=signal_length,
        sample_rate=sample_rate,
        leads=leads,
        heart_rate=heart_rate,
        preset=preset,
        noise_multiplier=noise_multiplier,
        impedance=impedance,
        p_multiplier=p_multiplier,
        t_multiplier=t_multiplier,
        voltage_factor=voltage_factor,
    )

    return ecg, segs, fids


def simulate_daubechies(signal_length: int = 10000, sample_rate: float = 1000, heart_rate: float = 70) -> npt.NDArray:
    """Generate an artificial (synthetic) ECG signal of a given duration and sampling rate.

    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_.

    Args:
        signal_length (int, optional): Length of the ECG signal. Defaults to 10000.
        sample_rate (float, optional): ECG sampling frequency. Defaults to 1000.
        heart_rate (float, optional): Heart rate in BPM. Defaults to 70.

    Returns:
        npt.NDArray: Synthetic ECG signal

    """
    duration = signal_length / sample_rate

    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = scipy.signal.daub(10)

    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac, num_heart_beats)

    # Change amplitude
    ecg = ecg * 10

    # Resample
    ecg = scipy.ndimage.zoom(ecg, signal_length / len(ecg))

    ecg = ecg[:signal_length]

    return ecg


def simulate_ecgsyn(
    signal_length: int = 10000,
    sample_rate: float = 256,
    leads: int = 12,
    heart_rate: float = 60,
    hr_std: float = 1,
    lfhfratio: float = 0.5,
    sfint: float = 512,
    ti: tuple[int] = (-70, -15, 0, 15, 100),
    ai: tuple[float] = (1.2, -5, 30, -7.5, 0.75),
    bi: tuple[float] = (0.25, 0.1, 0.1, 0.1, 0.4),
    gamma: npt.NDArray | None = None,
) -> npt.NDArray:
    """Simulate ECG using the ECGSYN algorithm.

    This function is a python translation of the matlab script by `McSharry & Clifford (2013)
    <https://physionet.org/content/ecgsyn>`_.

    Args:
        signal_length (int, optional): Length of the ECG signal. Defaults to 10000.
        sample_rate (float, optional): ECG sampling frequency. Defaults to 256.
        heart_rate (float, optional): Mean heart rate. Defaults to 60.
        hr_std (float, optional): Heart rate standard deviation. Defaults to 1.
        lfhfratio (float, optional): Low frequency high frequency ratio. Defaults to 0.5.
        sfint (float, optional): Internal sampling frequency. Defaults to 512.
        ti (tuple[int], optional): Time parameters. Defaults to (-70, -15, 0, 15, 100).
        ai (tuple[float], optional): Amplitude parameters. Defaults to (1.2, -5, 30, -7.5, 0.75).
        bi (tuple[float], optional): Width parameters. Defaults to (0.25, 0.1, 0.1, 0.1, 0.4).
        gamma (npt.NDArray|None, optional): Lead modification matrix. Defaults to None.

    Returns:
        tuple[list[npt.NDArray], list[npt.NDArray]]: ECG signals and results

    """
    if gamma is None:
        gamma = np.array(
            [
                [1, 0.1, 1, 1.2, 1],
                [2, 0.2, 0.2, 0.2, 3],
                [1, -0.1, -0.8, -1.1, 2.5],
                [-1, -0.05, -0.8, -0.5, -1.2],
                [0.05, 0.05, 1, 1, 1],
                [1, -0.05, -0.1, -0.1, 3],
                [-0.5, 0.05, 0.2, 0.5, 1],
                [0.05, 0.05, 1.3, 2.5, 2],
                [1, 0.05, 1, 2, 1],
                [1.2, 0.05, 1, 2, 2],
                [1.5, 0.1, 0.8, 1, 2],
                [1.8, 0.05, 0.5, 0.1, 2],
            ]
        )
    # END IF

    if not isinstance(ti, np.ndarray):
        ti = np.array(ti)
    if not isinstance(ai, np.ndarray):
        ai = np.array(ai)
    if not isinstance(bi, np.ndarray):
        bi = np.array(bi)

    duration = signal_length / sample_rate

    # Number of beats
    N = int(np.round(duration * (heart_rate / 60)))

    ti = ti * np.pi / 180

    # Adjust extrema parameters for mean heart rate
    hrfact = np.sqrt(heart_rate / 60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * bi
    ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2]) * ti

    # Check that sfint is an integer multiple of sfecg
    q = np.round(sfint / sample_rate)
    qd = sfint / sample_rate
    if q != qd:
        raise ValueError(
            "Internal sampling frequency (sfint) must be an integer multiple of the ECG sampling frequency"
            " (sfecg). Your current choices are: sfecg = " + str(sample_rate) + " and sfint = " + str(sfint) + "."
        )

    # Define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and respiratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # Calculate time scales for rr and total output
    sfrr = 1
    trr = 1 / sfrr
    rrmean = 60 / heart_rate
    n = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = _ecg_simulate_rrprocess(flo, fhi, flostd, fhistd, lfhfratio, heart_rate, hr_std, sfrr, n)

    # Upsample rr time series from 1 Hz to sfint Hz
    desired_length = int(np.round(len(rr0) * sfint / 1))
    rr = scipy.ndimage.zoom(rr0, desired_length / len(rr0))

    # Make the rrn time series
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg = 0
    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(np.round(tecg / dt))
        rrn[i:ip] = rr[i]
        i = ip
    Nt = ip

    # Integrate system using fourth order Runge-Kutta
    x0 = np.array([1, 0, 0.04])

    # tspan is a tuple of (min, max) which defines the lower and upper bound of t in ODE
    # t_eval is the list of desired t points for ODE
    # in Matlab, ode45 can accepts both tspan and t_eval in one argument
    Tspan = [0, (Nt - 1) * dt]
    t_eval = np.linspace(0, (Nt - 1) * dt, Nt)

    # Initialize results containers
    results = []
    signals = []

    # Multichannel modification (#625):
    # --------------------------------------------------
    # Loop over the twelve leads modifying ai in the loop to generate each lead's data
    # Because these are all starting at the same position, it may make sense to grab a random
    # segment within the series to simulate random phase and to forget the initial conditions

    for i in range(leads):
        gamma_row = gamma[i, :]
        result = scipy.integrate.solve_ivp(
            fun=functools.partial(_ecg_simulate_derivsecgsyn, rr=rrn, ti=ti, sfint=sfint, ai=gamma_row * ai, bi=bi),
            t_span=Tspan,
            y0=x0,
            t_eval=t_eval,
        )
        results.append(result)
        X0 = result.y  # get signal

        # downsample to required sfecg
        X = X0[:, np.arange(0, X0.shape[1], q).astype(int)]

        # Scale signal to lie between -0.4 and 1.2 mV
        z = X[2, :].copy()
        zmin = np.min(z)
        zmax = np.max(z)
        zrange = zmax - zmin
        z = (z - zmin) * 1.6 / zrange - 0.4

        signals.append(z)
    # END FOR

    signals = np.hstack(signals)

    signals = signals[:signal_length]
    return signals


def _ecg_simulate_derivsecgsyn(
    t: float, x: npt.NDArray, rr: npt.NDArray, ti: npt.NDArray, sfint: float, ai: npt.NDArray, bi: npt.NDArray
):
    """"""
    ta = math.atan2(x[1], x[0])
    r0 = 1
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2) / r0

    ip = np.floor(t * sfint).astype(int)
    w0 = 2 * np.pi / rr[min(ip, len(rr) - 1)]
    # w0 = 2*np.pi/rr[ip[ip <= np.max(rr)]]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * x[0] - w0 * x[1]
    dx2dt = a0 * x[1] + w0 * x[0]

    # matlab rem and numpy rem are different
    # dti = np.remainder(ta - ti, 2*np.pi)
    dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1 * (x[2] - zbase)

    dxdt = np.array([dx1dt, dx2dt, dx3dt])
    return dxdt


def _ecg_simulate_rrprocess(
    flo: float = 0.1,
    fhi: float = 0.25,
    flostd: float = 0.01,
    fhistd: float = 0.01,
    lfhfratio: float = 0.5,
    hrmean: float = 60,
    hrstd: float = 1,
    sfrr: float = 1,
    n: int = 256,
) -> npt.NDArray:
    """Simulate RR process using the ECGSYN algorithm.

    Args:
        flo (float, optional): Low frequency. Defaults to 0.1.
        fhi (float, optional): High frequency. Defaults to 0.25.
        flostd (float, optional): Low frequency standard deviation. Defaults to 0.01.
        fhistd (float, optional): High frequency standard deviation. Defaults to 0.01.
        lfhfratio (float, optional): Low frequency high frequency ratio. Defaults to 0.5.
        hrmean (float, optional): Mean heart rate. Defaults to 60.
        hrstd (float, optional): Heart rate standard deviation. Defaults to 1.
        sfrr (float, optional): RR sampling frequency. Defaults to 1.
        n (int, optional): Number of samples. Defaults to 256.

    Returns:
        npt.NDArray: RR time series.
    """
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1**2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2**2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate((Hw[0 : int(n / 2)], Hw[int(n / 2) - 1 :: -1]))
    Sw = (sfrr / 2) * np.sqrt(Hw0)

    ph0 = 2 * np.pi * np.random.uniform(size=int(n / 2 - 1))
    ph = np.concatenate([[0], ph0, [0], -np.flipud(ph0)])
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    xstd = np.std(x)
    ratio = rrstd / xstd
    return rrmean + x * ratio  # Return RR
