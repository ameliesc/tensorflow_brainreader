import matplotlib.pyplot as plt
from data import Data

data = Data(dataset = "Val", subject = "S1", roi = 1)
X_val = data.stimuli()
y_val = data.response()

data = Data(dataset = "Trn", subject = "S1", roi = 1)

X_trn = data.stimuli()
y_trn = data.stimuli()

plt.plot(y_val[0], y_val[1], '.')
