import os
import numpy as np
import matplotlib.pyplot as plt

# fig langa script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuri", exist_ok=True)


# matricea Fourier: F[k][n] = e^(-2 pi j w[k] n / N), 2j = 2 * i (unitatea imaginara)
def fourier(w, N):
    F = np.zeros((len(w), N), dtype=complex)
    for k in range(len(w)):
        for n in range(N):
            F[k][n] = np.exp(-2j * np.pi * w[k] * n / N)
    return F


# ex1
N = 8
F = fourier(np.arange(N), N)    # w = 0, 1, ..., 7

plt.figure(figsize=(6, 10))
plt.suptitle("ex1")
for m in range(N):
    plt.subplot(N, 1, m + 1)
    plt.plot(F[m].real, "o-", label="re")
    plt.plot(F[m].imag, "o-", label="im")
    plt.ylabel("m=" + str(m))
    if m == 0:
        plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figuri/ex1.pdf")

# F unitara: F^H * F = N * I (F^H = conjugata transpusa)
FH = np.conj(np.transpose(F))
print("ex1: F^H F = N I ?", np.allclose(np.dot(FH, F), N * np.eye(N)))


# ex2
f = 3
N = 1000
t = np.arange(N) / N                   # n / N: 1000 esant in [0, 1) s
x = np.cos(2 * np.pi * f * t)
y = x * np.exp(-2j * np.pi * t)        # infasurarea pe cercul unitate
k = 620                                # esant marcat cu rosu

w = [1, 2, 3, 6]                       # 3 = f
z = np.zeros((4, N), dtype=complex)
for i in range(4):
    z[i] = x * np.exp(-2j * np.pi * w[i] * t)

# fig1 static, culoare = distanta fata de origine
plt.figure(figsize=(10, 4))
plt.suptitle("ex2 fig1")
plt.subplot(1, 2, 1)
plt.scatter(t, x, c=np.abs(x), s=2)
plt.plot([t[k], t[k]], [0, x[k]], "r")
plt.plot(t[k], x[k], "ro")
plt.xlabel("t [s]")
plt.subplot(1, 2, 2)
plt.scatter(y.real, y.imag, c=np.abs(y), s=2)
plt.plot([0, y[k].real], [0, y[k].imag], "r")
plt.plot(y[k].real, y[k].imag, "ro")
plt.axis("square")
plt.xlabel("re")
plt.ylabel("im")
plt.tight_layout()
plt.savefig("figuri/ex2_fig1.pdf")

# fig2 static
plt.figure(figsize=(8, 8))
plt.suptitle("ex2 fig2")
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.scatter(z[i].real, z[i].imag, c=np.abs(z[i]), s=2)
    plt.plot(np.mean(z[i]).real, np.mean(z[i]).imag, "ko")    # centrul de masa
    plt.axis("square")
    plt.title("w=" + str(w[i]))
plt.tight_layout()
plt.savefig("figuri/ex2_fig2.pdf")

# animatii: pas = puncte noi pe cadru, pauza = timp intre cadre [s]
pas = 20
pauza = 0.01

# fig1 animat
plt.figure(figsize=(10, 4))
for i in range(pas, N + 1, pas):
    ti = t[:i]                         # primele i puncte
    xi = x[:i]
    yi = y[:i]
    plt.clf()
    plt.suptitle("ex2 fig1 anim")
    plt.subplot(1, 2, 1)
    plt.scatter(ti, xi, c=np.abs(xi), s=2, vmin=0, vmax=1)    # vmin, vmax: culori fixe
    plt.plot(t[i - 1], x[i - 1], "ro")                         # punctul curent
    plt.xlim(0, 1)
    plt.ylim(-1.1, 1.1)
    plt.subplot(1, 2, 2)
    plt.scatter(yi.real, yi.imag, c=np.abs(yi), s=2, vmin=0, vmax=1)
    plt.plot(y[i - 1].real, y[i - 1].imag, "ro")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis("square")
    plt.pause(pauza)

# fig2 animat
plt.figure(figsize=(8, 8))
for i in range(pas, N + 1, pas):
    plt.clf()
    plt.suptitle("ex2 fig2 anim")
    for j in range(4):
        zj = z[j][:i]
        plt.subplot(2, 2, j + 1)
        plt.scatter(zj.real, zj.imag, c=np.abs(zj), s=2, vmin=0, vmax=1)
        plt.plot(z[j][i - 1].real, z[j][i - 1].imag, "ro")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.axis("square")
        plt.title("w=" + str(w[j]))
    plt.pause(pauza)
# obs: punctele se deseneaza in sens orar (minusul din exponent)
# obs: doar la w = f desenul nu e simetric fata de origine, centrul de masa iese din 0


# ex3
N = 1000
t = np.arange(N) / N                   # fs = 1000 Hz, 1 s
x1 = np.cos(2 * np.pi * 10 * t)
x2 = 2 * np.cos(2 * np.pi * 25 * t)
x3 = 0.5 * np.cos(2 * np.pi * 60 * t)
x = x1 + x2 + x3
w = np.arange(0, 101)                  # w = 0..100 Hz, acopera 10, 25, 60

# v1: relatia (1) cu for-uri
X = np.zeros(len(w), dtype=complex)
for k in range(len(w)):
    s = 0
    for n in range(N):
        s = s + x[n] * np.exp(-2j * np.pi * w[k] * n / N)
    X[k] = s

# v2: matricea Fourier * vector
X2 = np.dot(fourier(w, N), x)
print("ex3: for-uri = matrice ?", np.allclose(X, X2))

plt.figure(figsize=(10, 4))
plt.suptitle("ex3")
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.xlabel("t [s]")
plt.subplot(1, 2, 2)
plt.stem(w, np.abs(X))
plt.xlabel("w [Hz]")
plt.ylabel("|X(w)|")
plt.tight_layout()
plt.savefig("figuri/ex3.pdf")
# obs: |X| != 0 doar la 10, 25, 60 Hz, cu inaltimea A * N / 2 = 500, 1000, 250

plt.show()
