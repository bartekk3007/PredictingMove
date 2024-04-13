
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
import random
import numpy as np
import time

start = time.time()

rozmiar = 10000
plansza = 3
czesc_rozmiaru = int(rozmiar * 4 / 5)

indeksy = range(rozmiar)
indeksy_wybrane = random.sample(indeksy, czesc_rozmiaru)
indeksy_pozostale = list(set(indeksy) - set(indeksy_wybrane))

inputsFile = []
labelsFile = []


def char_converter(znak):
    if znak == '-':
        return 0
    elif znak == 'X':
        return 1
    elif znak == 'O':
        return 2


def output_converter(array):
    indeks = array.index(1)
    return indeks

with open('DanePlanszyWszystkie.txt') as f:
    for _ in range(rozmiar):
        tab = []
        f.readline()
        for _ in range(plansza):
            linia = f.readline()
            tab.append(char_converter(linia[0]))
            tab.append(char_converter(linia[2]))
            tab.append(char_converter(linia[4]))
        f.readline()
        inputsFile.append(tab)
        linia_wyniku = f.readline()
        wynik = 3 * int(linia_wyniku[0]) + int(linia_wyniku[2])
        tab_outputu = []
        for i in range(plansza*plansza):
            if i == wynik:
                tab_outputu.append(1)
            else:
                tab_outputu.append(0)
        labelsFile.append(tab_outputu)

# Dane wejściowe
inputs = [inputsFile[indeks] for indeks in indeksy_wybrane]

# Odpowiednie etykiety (wyniki)
labels = [labelsFile[indeks] for indeks in indeksy_wybrane]

# Tworzenie modelu
model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))  # Warstwa ukryta z 64 neuronami
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trenowanie modelu
model.fit(inputs, labels, epochs=200)

# Testowanie modelu
test_inputs = [inputsFile[indeks] for indeks in indeksy_pozostale]
test_labels = [labelsFile[indeks] for indeks in indeksy_pozostale]
predictions = model.predict(test_inputs)
predicted_classes = np.argmax(predictions, axis=1)

end = time.time()
# Wyświetlanie wyników
licznik = 0
iter = 0
for i, input_data in enumerate(test_inputs):
    print("Dane wejściowe:", input_data)
    print("Przewidywany wynik:", predicted_classes[i])
    predykacja = predicted_classes.item(i)
    print("Przybliżony wynik:", predykacja)
    print("Prawidlowy wynik:", output_converter(test_labels[i]))
    if output_converter(test_labels[i]) == predykacja:
        print("Poprawnie przewidziano rezultat")
        licznik += 1
    iter += 1
    print("Stosunek poprawnie przewidzianych", licznik / iter)
    print()
print("Czas jaki zajal trening to", end-start)