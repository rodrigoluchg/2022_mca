import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics 		 import confusion_matrix, classification_report
from sklearn.utils           import class_weight

from tensorflow.keras.models     import Sequential, load_model
from tensorflow.keras.layers     import *
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.losses     import MeanSquaredError
from tensorflow.keras.metrics    import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

dataset_path  = 'datasets/dataset_final.csv'
output_folder = 'output'

def main():
    print("---------------------------")
    print("------------LSTM-----------")
    print("---------------------------")

    df = pd.read_csv(dataset_path)

    features      = ["CPU_USED_PCT", "MEM_USED_PCT", "SWAP_USED_PCT", "LOAD_AVERAGE_1_MIN", "RUN_QUEUE"]
    predict_class = "AVAILABILITY"
 
    x = df[features].values
    y = df[predict_class].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    lstm_internal_units     = 64 # quantidade de neuronios na camada de regressao
    dense_layer_size        = 8 # quantidade de neuronios na camada densa
    output_dense_layer_size = 1 # camada de saida, igual a quantidade de classes
    n_epochs                = 10 # quantiddade de vezes que o treinamento ira acontecer

    # Calculo do peso das clases (0 - Indisponivel, 1 - Disponivel)
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights = {0: weights[0], 1: weights[1]}

    print(f"PESOS: {weights}")

    # Montagem da rede neural
    model = Sequential()
    model.add(InputLayer((len(features), 1)))
    model.add(LSTM(lstm_internal_units))
    model.add(Dense(dense_layer_size, "relu")) # Retorna 0 se a entrada for negativa, senao retorna o proprio valor da entrada
    model.add(Dense(output_dense_layer_size, 'sigmoid'))   # Retorno binario (0,1)

    print(model.summary()) # Formato da Rede Neural

    model_bkp = 'lstm_model/'
    cp        = ModelCheckpoint(model_bkp, save_best_only=True) # Salva somente a melhor execucao

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), 'accuracy'])

    start = time()    
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=n_epochs, callbacks=[cp], class_weight=weights)
    print(f"DURACAO TREINAMENTO: {time() - start}")

    model = load_model(model_bkp)

    start = time()
    y_pred = model.predict(X_test).flatten()
    y_pred = np.array(list(map(lambda x: 0 if x < 0.5 else 1, y_pred)))
    print(f"DURACAO PREDICAO: {time() - start}")

    print("\n---------------------------------")
    print("Matriz de confusao")
    print(confusion_matrix(y_test, y_pred))

    print("---------------------------------")
    print("Cross tab")
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    print("---------------------------------")
    print("Diagnostico de classificacao")
    print(classification_report(y_test,y_pred))

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicao (classificacao)', fontsize=18)
    plt.ylabel('Valor real', fontsize=18)
    plt.title('Matriz de confusao - LSTM', fontsize=18)
    plt.show()
    fig.savefig("output/figs/lstm_confusion_matrix.png")

if __name__ == "__main__":
    main()