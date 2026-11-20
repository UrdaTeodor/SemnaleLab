import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

# fig langa script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuri", exist_ok=True)


# DFT ca in lab3, cu for pe k: X[k] = suma x[n] *e^(-2 pi j k n / N)
def dft(x):
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))
    return X


# FFT recursiv N = putere a lui 2
def fft(x):
    N = len(x)
    if N == 1:
        return x
    par = np.zeros(N // 2)
    impar = np.zeros(N // 2)
    for i in range(N // 2):
        par[i] = x[2 * i]
        impar[i] = x[2 * i + 1]
    P = fft(par)
    I = fft(impar)
    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        w = np.exp(-2j * np.pi * k / N) * I[k]
        X[k] = P[k] + w
        X[k + N // 2] = P[k] - w
    return X


# cat dureaza functie(x) [s]
def timp(functie, x):
    start = time.perf_counter()
    functie(x)
    return time.perf_counter() - start


# ca Figura 2: fiecare semnal din f, cu esantioanele lui luate la fs
def desen(f, fs, titlu, fisier):
    t = np.linspace(0, 0.05, 5000)          # aproape continuu
    ts = np.arange(0, 0.05, 1 / fs)         # momentele de esantionare
    plt.figure(figsize=(8, 6))
    plt.suptitle(titlu)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, np.sin(2 * np.pi * f[i] * t))
        plt.plot(ts, np.sin(2 * np.pi * f[i] * ts), "o")
        plt.ylabel(str(f[i]) + " Hz")
    plt.tight_layout()
    plt.savefig(fisier)


# ex1
Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
t_dft = np.zeros(len(Ns))
t_fft = np.zeros(len(Ns))
t_np = np.zeros(len(Ns))
for i in range(len(Ns)):
    x = np.random.rand(Ns[i])
    t_dft[i] = timp(dft, x)
    t_fft[i] = timp(fft, x)
    t_np[i] = timp(np.fft.fft, x)
print("ex1: fft corect:", np.allclose(fft(x), np.fft.fft(x)))

plt.figure()
plt.suptitle("ex1")
plt.semilogy(Ns, t_dft, "o-", label="dft")
plt.semilogy(Ns, t_fft, "o-", label="fft")
plt.semilogy(Ns, t_np, "o-", label="np.fft")
plt.xlabel("N")
plt.ylabel("t [s]")
plt.legend()
plt.savefig("figuri/ex1.pdf")
# obs: dft creste ~N^2, fft ~N log N; np.fft e cel mai rapid (scris in C)


# ex2: 260, 140, 20 Hz difera prin multipli de 120 Hz
f = [260, 140, 20]
desen(f, 120, "ex2: fs=120 < 2*260 (aliere)", "figuri/ex2.pdf")
# obs: toate 3 trec prin aceleasi esantioane, deci nu stim care a fost semnalul initial


# ex3
desen(f, 600, "ex3: fs=600 > 2*260 (fara aliere)", "figuri/ex3.pdf")
# obs: esantioanele difera, deci semnalul de 260 Hz se poate reconstrui fara ambiguitate


# ex4: contrabas intre 40 si 200 Hz
print("ex4: fs >", 2 * 200, "Hz")


# ex5: spectrograma
# se pot deosebi


# ex6
rate, x = scipy.io.wavfile.read("vocale.wav")
N = len(x)
L = N // 100                                # un grup = 1% din semnal
pas = L // 2                                # suprapunere 50%
nr = (N - L) // pas + 1                     # cate grupuri incap

S = np.zeros((L // 2, nr))
for i in range(nr):
    grup = x[i * pas:i * pas + L]           # x[i * pas], ..., x[i * pas + L - 1]
    X = np.abs(np.fft.fft(grup))
    for k in range(L // 2):                 # doar frecventele pozitive
        S[k][i] = X[k]                      # FFT-ul grupului i pe coloana i

plt.figure(figsize=(10, 4))
plt.suptitle("ex6")
plt.imshow(20 * np.log10(S + 1), origin="lower", aspect="auto", cmap="inferno",
           extent=[0, N / rate, 0, rate / 2])    # axele: t [s], f [Hz]
plt.colorbar(label="dB")
plt.ylim(0, 8000)                           # vocea e sub ~8 kHz
plt.xlabel("t [s]")
plt.ylabel("f [Hz]")
plt.savefig("figuri/ex6.pdf")
#~600-1200 Hz, e: ~500 Hz si ~2000 Hz, i: ~250 Hz si ~3000-3700 Hz;
#      o: ~500-900 Hz; u: cel mai jos, ~250-700 Hz


# ex7: SNR_dB = P_semnal_dB - P_zgomot_dB
P_semnal = 90
SNR = 80
P_zgomot = P_semnal - SNR
print("ex7: P_zgomot =", P_zgomot, "dB =", 10 ** (P_zgomot / 10), "* P0")

plt.show()
