import os
import json
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np

from sklearn.neighbors 		 import KNeighborsClassifier
from sklearn.metrics 		 import confusion_matrix, classification_report
from sklearn 				 import metrics
from scipy 					 import stats
from sklearn                 import preprocessing
from sklearn                 import utils
from sklearn.model_selection import train_test_split

from time import time

# Comentar um dos servers para obter metricas
# dataset_path  = 'datasets/estatisticas-luch.xlsx - v2.csv'
dataset_path  = 'datasets/dataset_final.csv'
# dataset_path  = 'datasets/dataset_final_strip2.csv'
output_folder = 'output'

def plot_double(x, x_label, y1, y1_label, y2, y2_label, show_plot=False):
    fig, ax1 = plt.subplots()
    ax1.set_title('{} x {} [%]'.format(y1_label, y2_label), size=25)
    ax1.set_xlabel(x_label, size=25)
    ax1.set_ylabel('{} [%]'.format(y1_label), color='tab:blue', size=25)
    ax1.plot(y1, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    plt.grid(color='blue', which='both')
    plt.xticks(rotation=0)

    ax2 = ax1.twinx()

    ax2.set_ylabel('{} [%]'.format(y2_label), color='tab:orange', size=25)
    ax2.plot(y2, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    plt.grid(color='orange', which='both')

    plt.xticks(rotation=0)

    fig.tight_layout()

    if show_plot:
        plt.show()

    fig.savefig('{}/figs/{}_{}.png'.format(output_folder, y1_label, y2_label), dpi=fig.dpi)


# Procedimento que recebe o dataset como parametro
def get_statistics(df):
    statistics = {}
    
# Varre todas as métricas no array    
    for column in df.columns[1:]:
        statistics[column] = df[column].describe().to_dict()

#Retorna todas as estatisticas para um arquivo
    return statistics

def to_json_file(dict_data, file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_data, fp, indent=2)

def get_boxplot(data, label, show_plot=False):
    fig, ax = plt.subplots()

    sns.set_theme(style="whitegrid")
    ax.axes.set_title('Boxplot - {}'.format(label),fontsize=25)
    ax.tick_params(axis='x', labelsize=15)
    ax.xaxis.grid(True)
    ax = sns.boxplot(x=data).set(xlabel='{} [%]'.format(label))

    if show_plot:
        plt.show()

    fig.savefig('{}/figs/{}_boxplot.png'.format(output_folder, label), dpi=fig.dpi)

def get_histogram(data, label, show_plot=False):
    fig, ax = plt.subplots()

    sns.set_theme(style="whitegrid")
    ax = sns.histplot(x=data)
    ax.axes.set_title('Histograma - {}'.format(label),fontsize=25)
    ax.set_xlabel('{} [%]'.format(label),fontsize=25)
    ax.set_ylabel('Quantidade de medições',fontsize=25)
    ax.tick_params(axis='x', labelsize=15)

    if show_plot:
        plt.show()

    fig.savefig('{}/figs/{}_hist.png'.format(output_folder, label), dpi=fig.dpi)

def get_graphics(df):
    for k in list(df.columns.values):
        if k == "DT_COLETA": continue

        # Gera boxplots
        get_boxplot(df[k], k, False)

        # Gera histogramas
        get_histogram(df[k], k, False)

    # Plota graficos de comparacao
    plot_double(df['DT_COLETA'], 'Amostras', df['MEM_USED_PCT'], 'Memoria', df['CPU_USED_PCT'], 'CPU', False)
    plot_double(df['DT_COLETA'], 'Amostras', df['MEM_USED_PCT'], 'Memoria', df['SWAP_USED_PCT'], 'SWAP', False)
    plot_double(df['DT_COLETA'], 'Amostras', df['AVAILABILITY'], 'Disponibilidade', df['CPU_USED_PCT'], 'CPU', False)
    plot_double(df['DT_COLETA'], 'Amostras', df['AVAILABILITY'], 'Disponibilidade', df['MEM_USED_PCT'], 'Memoria', False)
    plot_double(df['DT_COLETA'], 'Amostras', df['MEM_USED_PCT'], 'Memoria', df['SWAP_USED_PCT'], 'SWAP', False)

def get_correlation(data1_df, data2_df):
    data1_list = data1_df.tolist()
    data2_list = data2_df.tolist()

    correlation, _ = stats.pearsonr(data1_list, data2_list)

    return correlation

def main():
    print("----------------------")
    print("---------K-NN---------")
    print("----------------------")

    pre_execution = True

    # Efetua a leitura do dataset
    df = pd.read_csv(dataset_path)

    # Cria os diretorios caso nao existam
    if not os.path.exists('{}/figs'.format(output_folder)):
        os.makedirs('{}/figs'.format(output_folder))

    if pre_execution:
        # Obtem graficos de comparacao, boxplots e histogramas
        get_graphics(df)

        # Obtem dados estaticos (min, max, media, mediana, etc)
        statistics = get_statistics(df)

        # Verifica correlacao entre as metricas
        for k in statistics.keys():
            if k == "DT_COLETA" or k == "AVAILABILITY": continue

            corr = get_correlation(df["AVAILABILITY"], df[k])

            statistics[k]["corr_availability"] = corr

        # Salva dados estatisticos em um arquivo json
        to_json_file(statistics, '{}/statistics.json'.format(output_folder))

###########################################################################################
    # df_without_nan = df.dropna()
    df_without_nan = df.copy()

    # x = df_without_nan[["CPU_USED_PCT", "MEM_USED_PCT", "SWAP_USED_PCT", "IO_BUSY", "LOAD_AVERAGE_1_MIN", "RUN_QUEUE"]].values
    x = df_without_nan[["CPU_USED_PCT", "MEM_USED_PCT", "SWAP_USED_PCT", "LOAD_AVERAGE_1_MIN", "RUN_QUEUE"]].values
    y = df_without_nan["AVAILABILITY"].values

    neighbors         = 25
    neighbors         = np.arange(1,neighbors)
    test_accuracy_knn = np.empty(len(neighbors))

    knn_accuracy_list         = []
    k_variation 	          = []
    train_duration_list       = []
    train_duration_List_by_k  = []

    max_tries = 4
    i         = 0
    while i < max_tries:
        print(f"INCIANDO LACO {i}\n\n")

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        train_duration_list.append([])

        for j,k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
        
            start = time()
            #divide o modelo
            knn.fit(X_train, y_train)	
            train_duration_list[i].append(time() - start)

            # Calcula a acuracia dos dados de teste
            test_accuracy_knn[j] = knn.score(X_test, y_test)

        knn_accuracy_list.append(max(test_accuracy_knn))
        k_variation.append(np.argmax(test_accuracy_knn) + 1)
        train_duration_List_by_k.append(np.argmax(test_accuracy_knn))

        i += 1

    duration_best_fit = []

    i = 0
    while i < len(train_duration_list):
        duration_best_fit.append(train_duration_list[i][train_duration_List_by_k[i]])

        i += 1

    k_variation      = np.array(k_variation)
    k_variation_mean = np.mean(k_variation)

    knn_accuracy_list   = np.array(knn_accuracy_list)
    knn_accuracy_mean   = np.mean(knn_accuracy_list)
    train_duration_mean = np.mean(np.array(duration_best_fit))
    
    print("Acuracia media: " + str(knn_accuracy_mean))
    print("Duracao media do treinamento [s]: " + str("%.4f" % train_duration_mean))

    print("-------Vizinhos-------")
    print("Quantidade de vizinhos com melhor acuracia em cada execucao: " + str(k_variation))
    print("Media: " + str(k_variation_mean))
    print("Min:   " + str(np.amin(k_variation)))
    print("Max:   " + str(np.amax(k_variation)))

    most_repeated_k = np.bincount(k_variation).argmax()
    print("numero de vizinhos mais repetido: " + str(most_repeated_k))

    fig, ax1 = plt.subplots()
    ax1.set_title('k-NN variando numero de K', size=20)
    ax1.plot(neighbors, test_accuracy_knn, label='Precisao K-NN')
    ax1.set_xlabel('Numero de vizinhos proximos', size=25)
    ax1.set_ylabel('Acuracia', size=25)
    plt.grid(visible=True, axis='both')
    plt.show()

    fig.tight_layout()

    fig.savefig('{}/figs/{}_{}.png'.format(output_folder, "knn", "variation"), dpi=fig.dpi)

###########################################################################################
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=most_repeated_k)
    start = time()
    knn_classifier.fit(X_train, y_train)
    print(f"DURACAO TREINAMENTO: {time() - start}")

    start = time()
    knn_y_pred = knn_classifier.predict(X_test)
    print(f"DURACAO PREDICAO: {time() - start}")


    print("---------------------------------")
    print("Matriz de confusao")
    print(confusion_matrix(y_test, knn_y_pred))

    print("---------------------------------")
    print("Cross tab")
    print(pd.crosstab(y_test, knn_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    print("---------------------------------")
    print("Diagnostico de classificacao")
    print(classification_report(y_test,knn_y_pred))

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=knn_y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predicao (classificacao)', fontsize=18)
    plt.ylabel('Valor real', fontsize=18)
    plt.title('Matriz de confusao - KNN', fontsize=18)
    plt.show()
    fig.savefig("output/figs/knn_confusion_matrix.png")

if __name__ == "__main__":
    main()