import tensorflow as tf
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Rete neurale fondamentale: 1 strato, 1 neurone

# Questo comando crea un modello sequenziale, ovvero una struttura semplice di rete neurale in cui i layer (strati) 
# sono empilati uno dopo l'altro.
model = tf.keras.Sequential([

    # Define the input shape => La rete si aspetta un input monodimensionale, con un singolo valore
    tf.keras.Input(shape=(1,)), 

    # Add a Dense layer => Layer completamente connesso (denso) => units=1 tipico dei modelli di regressione
    # perché ci si aspetta come output un valore continuo
    tf.keras.layers.Dense(units=1)
]) 

# Compilazione del modello
# sgd sta per Stochastic Gradient Descent =>  aggiorna i pesi del modello in modo incrementale, 
# calcolando la derivata della perdita rispetto ai pesi su piccole porzioni (batch) dei dati invece che sull’intero dataset. 
# È particolarmente utile per grandi dataset, poiché consente un addestramento più veloce.
# mean_squared_error (errore quadratico medio) è una funzione di perdita comunemente usata nei problemi di regressione.
# Calcola la media dei quadrati delle differenze tra le previsioni del modello e i valori effettivi
model.compile(optimizer='sgd', loss='mean_squared_error')

# Addestramento del modello. Un’epoca è un ciclo completo attraverso tutto il dataset di addestramento.
# Quando specifichi epochs=500, stai dicendo al modello di scorrere l'intero dataset 500 volte durante l'addestramento.

# Stampo una prediction
print(f"""
      
Numero predetto dal modello: {model.predict(np.array([10.0]), verbose=0).item():.5f}

Hello World, Thx :)""")
