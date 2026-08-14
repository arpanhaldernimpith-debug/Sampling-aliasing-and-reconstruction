# Sampling-aliasing-and-reconstruction
import numpy as np
import matplotlib.pyplot as plt

f1, f2, A2 = 5.0, 12.0, 0.5
fs_cases = [40.0, 24.0, 18.0]
t = np.arange(-1.0, 1.0005, 0.0005)
x = np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)

def sinc_reconstruct(ts, xs, tq):
    y = np.zeros_like(tq)
    for i in range(0, len(tq), 2500):
        q = tq[i:i+2500]
        z = (q[:, None] - ts[None, :]) / (ts[1]-ts[0])
        y[i:i+2500] = np.sum(xs[None, :] * np.sinc(z), axis=1)
    return y

def alias_frequency(f, fs):
    return abs(((f + fs/2) % fs) - fs/2)

for fs in fs_cases:
    Ts = 1/fs
    n = np.arange(np.ceil(-1/Ts), np.floor(1/Ts)+1, dtype=int)
    ts = n*Ts
    xs = np.sin(2*np.pi*f1*ts) + A2*np.sin(2*np.pi*f2*ts)
    xhat = sinc_reconstruct(ts, xs, t)

    rmse = np.sqrt(np.mean((xhat-x)**2))
    print(f"fs={fs:g} Hz, Nyquist={fs/2:g} Hz")
    print(f"5 Hz aliases to {alias_frequency(5,fs):g} Hz")
    print(f"12 Hz aliases to {alias_frequency(12,fs):g} Hz")
    print(f"RMSE={rmse:.6f}")

    plt.figure(figsize=(10,5))
    plt.plot(t, x, label="Reference")
    plt.plot(t, xhat, label="Sinc reconstruction")
    plt.stem(ts, xs, linefmt="C2-", markerfmt="C2o", basefmt=" ", label="Samples")
    plt.xlim(-0.5,0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Sampling and reconstruction, fs={fs:g} Hz")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
