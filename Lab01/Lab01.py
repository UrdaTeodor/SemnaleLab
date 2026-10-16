from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURI = Path(__file__).parent / "figuri"
FIGURI.mkdir(exist_ok=True)


def salveaza(fig, nume):
    fig.savefig(FIGURI / f"{nume}.pdf")

# fs1 = 400

# T = 1/fs1

# indici = np.arange(1600)
# tn = indici * T
# #1b
# x0 = np.linspace(0, 0.1, 100) 
# y0 = np.cos(520 * np.pi * x0 + np.pi/3.0)
# x1 = np.linspace(0, 0.1, 100)
# y1 = np.cos(280 * np.pi * x1 - np.pi/3.0)
# x2 = np.linspace(0, 0.1, 100)
# y2 = np.cos(120 * np.pi * x2 + np.pi/3.0)

# xn = np.cos(520 * np.pi * tn + np.pi/3.0)
# yn = np.cos(280 * np.pi * tn - np.pi/3.0)
# zn = np.cos(120 * np.pi * tn + np.pi/3.0)


# n = 3  # numarul de subploturi
# fig, axs = plt.subplots(n)
# fig.suptitle('2a')
# axs[0].plot(x0, y0)
# axs[1].plot(x1, y1)
# axs[2].plot(x2, y2)


# axs[0].stem(tn, xn)
# axs[1].stem(tn, yn)
# axs[2].stem(tn, zn)
# salveaza(fig, "ex2a")


f = 400                            # frecventa sinusoidei [Hz]
t = np.linspace(0, 0.01, 1600)     # 1600 de esantioane in 0.01 s (4 perioade)
x = np.sin(2 * np.pi * f * t)

fig, ax = plt.subplots()
fig.suptitle('2a')
ax.plot(t, x)
ax.set_xlabel('t [s]')
salveaza(fig, "ex2a")

f = 800
t = np.linspace(0, 3, 3 * 16000)
x = np.sin(2 * np.pi * f * t)

fig, ax = plt.subplots()
fig.suptitle('2b')
ax.plot(t, x)
ax.set_xlim([0, 0.005])
salveaza(fig, "ex2b")

f = 240
t = np.linspace(0, 0.01, 1000)
x = np.mod(f * t, 1)

fig, ax = plt.subplots()
fig.suptitle('2c')
ax.plot(t, x)
salveaza(fig, "ex2c")

f = 300
t = np.linspace(0, 0.01, 1000)
x = np.sign(np.sin(2 * np.pi * f * t))

fig, ax = plt.subplots()
fig.suptitle('2d')
ax.plot(t, x)
salveaza(fig, "ex2d")

I = np.random.rand(128, 128)

fig, ax = plt.subplots()
fig.suptitle('2e')
ax.imshow(I)
salveaza(fig, "ex2e")

I = np.zeros((128, 128))
I[32:96, 32:96] = np.ones((64, 64))

fig, ax = plt.subplots()
fig.suptitle('2f')
ax.imshow(I)
salveaza(fig, "ex2f")

print("3a:", 1 / 2000, "s")
print("3b:", 2000 * 3600 * 4 // 8, "bytes")






plt.show()
