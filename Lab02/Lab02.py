import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice

# fig + wav langa script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuri", exist_ok=True)


# ex1
A = 2
f = 5
phi = np.pi / 3
t = np.linspace(0, 1, 1000)    # 1000 val in [0, 1] s

# np.sin, np.cos: pe fiecare elem din t
x = A * np.sin(2 * np.pi * f * t + phi)
y = A * np.cos(2 * np.pi * f * t + phi - np.pi / 2)    # cos(u - pi/2) = sin(u)

plt.figure()
plt.suptitle("ex1")
plt.subplot(2, 1, 1)    # grila 2x1, subplot 1
plt.plot(t, x)
plt.title("sin")
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.title("cos")
plt.tight_layout()
plt.savefig("figuri/ex1.pdf")


# ex2
f = 5
t = np.linspace(0, 1, 1000)
phi = [0, np.pi / 4, np.pi / 2, np.pi]
lbl = ["0", "pi/4", "pi/2", "pi"]

plt.figure()
plt.suptitle("ex2a")
for i in range(4):
    plt.plot(t, np.sin(2 * np.pi * f * t + phi[i]), label="phi=" + lbl[i])
plt.legend()
plt.savefig("figuri/ex2a.pdf")

# SNR = ||x||^2 / (gamma^2 * ||z||^2)   =>   gamma = ||x|| / (sqrt(SNR) * ||z||)
x = np.sin(2 * np.pi * f * t)
z = np.random.normal(0, 1, len(t))    # N(0, 1)
snr = [0.1, 1, 10, 100]

plt.figure()
plt.suptitle("ex2b")
for i in range(4):
    gamma = np.linalg.norm(x) / (np.sqrt(snr[i]) * np.linalg.norm(z))
    plt.subplot(4, 1, i + 1)
    plt.plot(t, x + gamma * z)
    plt.title("SNR=" + str(snr[i]))
plt.tight_layout()
plt.savefig("figuri/ex2b.pdf")


# ex3
# 2a-2d din lab1, refacute la fs=44100, 1 s fiecare (2a avea 0.01 s, prea scurt)
fs = 44100
t = np.arange(0, 1, 1 / fs)    # 0, 1/fs, 2/fs, ... (1 s)

x2a = np.sin(2 * np.pi * 400 * t)
x2b = np.sin(2 * np.pi * 800 * t)
x2c = np.mod(240 * t, 1)                      # saw
x2d = np.sign(np.sin(2 * np.pi * 300 * t))    # square
s = [x2a, x2b, x2c, x2d]

for i in range(4):
    sounddevice.play(0.3 * s[i], fs)    # 0.3 = vol
    sounddevice.wait()                  # asteapta

scipy.io.wavfile.write("ex3.wav", fs, x2a)
rate, x = scipy.io.wavfile.read("ex3.wav")
print("ex3: n =", len(x), "fs =", rate)


# ex4
t = np.linspace(0, 0.05, 2000)
x = np.sin(2 * np.pi * 100 * t)    # sin 100 Hz
y = 2 * np.mod(60 * t, 1) - 1      # saw 60 Hz, in [-1, 1]

plt.figure()
plt.suptitle("ex4")
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("sin")
plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.title("saw")
plt.subplot(3, 1, 3)
plt.plot(t, x + y)
plt.title("sin+saw")
plt.tight_layout()
plt.savefig("figuri/ex4.pdf")


# ex5
fs = 44100
t = np.arange(0, 1, 1 / fs)
x = np.sin(2 * np.pi * 300 * t)
y = np.sin(2 * np.pi * 900 * t)

sounddevice.play(0.3 * np.append(x, y), fs)    # x apoi y
sounddevice.wait()
# obs: intai ton grav (300 Hz), apoi mai inalt (900 Hz): f mai mare = sunet mai inalt


# ex6
fs = 100
n = np.arange(0, 30)    # n = 0, 1, ..., 29
f = [fs / 2, fs / 4, 0]
lbl = ["f=fs/2", "f=fs/4", "f=0"]

plt.figure()
plt.suptitle("ex6")
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.stem(n, np.sin(2 * np.pi * f[i] * n / fs))
    plt.ylim(-1.1, 1.1)
    plt.title(lbl[i])
plt.tight_layout()
plt.savefig("figuri/ex6.pdf")
# obs fs/2: sin(pi*n) = 0, esantioanele cad fix in zerouri
# obs fs/4: 0, 1, 0, -1, ... (4 esant/perioada)
# obs 0 Hz: sin(0) = 0, constant


# ex7
fs = 1000
f = 50
t = np.arange(0, 0.1, 1 / fs)
x = np.sin(2 * np.pi * f * t)

xa = x[0::4]    # x[0], x[4], x[8], ... (din 4 in 4)
ta = t[0::4]
xb = x[1::4]    # x[1], x[5], x[9], ...
tb = t[1::4]

plt.figure()
plt.suptitle("ex7")
plt.subplot(3, 1, 1)
plt.stem(t, x)
plt.title("x, fs=1000")
plt.subplot(3, 1, 2)
plt.stem(ta, xa)
plt.title("x[0::4], fs=250")
plt.subplot(3, 1, 3)
plt.stem(tb, xb)
plt.title("x[1::4], fs=250")
plt.tight_layout()
plt.savefig("figuri/ex7.pdf")
# obs a: tot 50 Hz, dar 5 esant/perioada in loc de 20
# obs b: acelasi sin, decalat cu 1 ms (phi=pi/10)


# ex8
a = np.linspace(-np.pi / 2, np.pi / 2, 1000)
sin = np.sin(a)
taylor = a
pade = (a - 7 * a * a * a / 60) / (1 + a * a / 20)
err_taylor = np.abs(sin - taylor)
err_pade = np.abs(sin - pade)

plt.figure()
plt.suptitle("ex8a")
plt.subplot(2, 1, 1)
plt.plot(a, sin, label="sin")
plt.plot(a, taylor, label="taylor")
plt.plot(a, pade, "--", label="pade")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(a, err_taylor, label="err taylor")
plt.plot(a, err_pade, label="err pade")
plt.legend()
plt.tight_layout()
plt.savefig("figuri/ex8a.pdf")

plt.figure()
plt.suptitle("ex8b")
plt.semilogy(a, err_taylor, label="err taylor")
plt.semilogy(a, err_pade, label="err pade")
plt.legend()
plt.savefig("figuri/ex8b.pdf")

print("ex8: err max taylor =", np.max(err_taylor))
print("ex8: err max pade =", np.max(err_pade))
# obs: taylor ok doar langa 0, pade ok pe tot intervalul

plt.show()
