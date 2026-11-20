import os
import numpy as np
import matplotlib.pyplot as plt

# fig langa script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figuri", exist_ok=True)

# Train.csv: ID, Datetime, Count -> pastram doar Count (coloana 2)
tabel = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
N = len(tabel)
x = np.zeros(N)
for i in range(N):
    x[i] = tabel[i][2]


# a: cate un esantion pe ora
fs = 1 / 3600
print("a: fs =", fs, "Hz")

# b
print("b:", N, "ore =", N / 24, "zile =", N / 24 / 365, "ani")

# c: esantionat corect si optim => fs = 2 * f_max
print("c: f_max =", fs / 2, "Hz (perioada de 2 ore)")


# d
f = fs * np.arange(N // 2) / N              # frecventa pt fiecare X[k], k < N/2
X = np.fft.fft(x)
X = np.abs(X / N)
X = X[:N // 2]                              # doar jumatate, spectrul e simetric

plt.figure()
plt.suptitle("d")
plt.plot(f, X)
plt.xlabel("f [Hz]")
plt.ylabel("|X(f)|")
plt.savefig("figuri/d.pdf")


# e: X[0] = media semnalului; daca e != 0 avem componenta continua
print("e: X[0] = media =", X[0])
# obs: da, media ~139 != 0, deci are componenta continua; o eliminam scazand media
y = x - np.mean(x)                          # semnalul fara componenta continua
Y = np.abs(np.fft.fft(y) / N)
Y = Y[:N // 2]

plt.figure()
plt.suptitle("e (fara DC)")
plt.plot(f, Y)
plt.xlabel("f [Hz]")
plt.ylabel("|X(f)|")
plt.savefig("figuri/e.pdf")


# f: primele 4 maxime din |X| (fara DC)
Yc = np.copy(Y)
for i in range(4):
    k = np.argmax(Yc)                       # indicele maximului
    perioada = 1 / f[k] / 3600              # [ore]
    print("f: k =", k, "f =", f[k], "Hz, perioada =", perioada, "ore =", perioada / 24, "zile")
    Yc[k] = 0                               # ca sa gasim urmatorul maxim
# obs: k = 1, 2, 3 (762, 381, 254 zile) vin din trendul crescator, scris de FFT ca sinusoide
#      foarte lente (k = 2 ~ 1 an); k = 762 (24 ore) e ciclul zi/noapte;
#      urmeaza k = 109 (~7 zile), ciclul saptamanal (weekend cu trafic mic)


# g: luni = ziua de dupa weekend, iar weekendul = 2 zile la rand cu cel mai putin trafic
# (presupunem ca primul esantion e la ora 00:00)
total = np.zeros(7)                         # traficul total pe fiecare zi a saptamanii
for d in range(N // 24):                    # ziua d
    for h in range(24):                     # ora h din ziua d
        total[d % 7] = total[d % 7] + x[d * 24 + h]

sambata = 0
for i in range(7):
    if total[i] + total[(i + 1) % 7] < total[sambata] + total[(sambata + 1) % 7]:
        sambata = i
luni = (sambata + 2) % 7

start = luni * 24                           # primul esantion de luni
while start <= 1000:
    start = start + 7 * 24
print("g: start =", start)
# obs: verificat cu coloana Datetime: esantionul 1056 = luni 08-10-2012, ora 00:00

plt.figure(figsize=(10, 4))
plt.suptitle("g: o luna de trafic, de luni")
plt.plot(np.arange(start, start + 30 * 24), x[start:start + 30 * 24])
plt.xlabel("esantion (ora)")
plt.ylabel("nr masini")
plt.savefig("figuri/g.pdf")


# h: data de start, doar din semnal
# - ziua saptamanii o aflam ca la g (weekendul are trafic mic)
#-pentru data cautam zile cu trafic anormal de mic care se repeta anual
# neajunsuri: nu stim tara (deci nici sarbatorile), zgomotul si trendul pot masca o sarbatoare;
# acuratetea depinde de cati ani acopera semnalul si cat de clar ies sarbatorile in evidenta


# i: filtru trece-jos in frecventa: X[k] = 0 pt frecventele peste f_taiere
f_taiere = 1 / (6 * 3600)                   # pastram perioadele >= 6 ore
X = np.fft.fft(x)
for k in range(N):
    fk = min(k, N - k) * fs / N             # frecventa lui X[k] (X[N - k] e perechea lui X[k])
    if fk > f_taiere:
        X[k] = 0
x_filtrat = np.real(np.fft.ifft(X))
# obs: in spectru (fig e) peste 1/(6 ore) nu mai sunt varfuri, doar zgomot; sub raman trendul,
#      ciclul de 7 zile, cel de 24 ore si armonicele lui (12, 8, 6 ore)

plt.figure(figsize=(10, 4))
plt.suptitle("i: filtrat (perioade >= 6 ore)")
plt.plot(np.arange(start, start + 30 * 24), x[start:start + 30 * 24], label="original")
plt.plot(np.arange(start, start + 30 * 24), x_filtrat[start:start + 30 * 24], label="filtrat")
plt.xlabel("esantion (ora)")
plt.ylabel("nr masini")
plt.legend()
plt.savefig("figuri/i.pdf")

plt.show()
