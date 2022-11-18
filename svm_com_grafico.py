from calendar import c
import os
import json
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np

from sklearn.neighbors 		 import KNeighborsClassifier
from sklearn.svm 			 import SVC  
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

def get_statistics(df):
    statistics = {}
    
    for column in df.columns[1:]:
        statistics[column] = df[column].describe().to_dict()

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
    print("-------------------------")
    print("-----------SVM-----------")
    print("-------------------------")

    pre_execution = False

    # Efetua a leitura do dataset
    df = pd.read_csv(dataset_path)

    # Cria os diretorios caso nao existam
    if not os.path.exists('{}/figs'.format(output_folder)):
        os.makedirs('{}/figs'.format(output_folder))

    if pre_execution:
        # Obtem graficos de comparacao, boxplots e histogramas
        # get_graphics(df)

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

    # print(df_without_nan.head())
    # print("--------------------------------------------------------")

#   x = df_without_nan[["CPU_USED_PCT", "MEM_USED_PCT"]].values
    x = df_without_nan[["SWAP_USED_PCT", "LOAD_AVERAGE_1_MIN"]].values
    
    y = df_without_nan["AVAILABILITY"].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # linear, poly, rbf, sigmoid, precomputed
    kernel_func = 'linear'
    
    # kernel coefficient for rbf, poly and sigmoid. Deve ser maior que 0
    # tabem definido como "auto" e "scale"(default). Se for "auto", considera-se gamma = 1/numero de features
    gamma_value = 'scale'

    # Regularization parameter - deve ser positivo
    C_value = 1.0

    svm_classifier = SVC(kernel=kernel_func, gamma=gamma_value, C=C_value)

    start = time()
    svm_classifier.fit(X_train, y_train)
    print(f"DURACAO TREINAMENTO: {time() - start}")

    print("GERANDO GRAFICO COM AS CLASSES")

    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt

    # Plotting decision regions
    plot_decision_regions(X_train, y_train, clf=svm_classifier, legend=2)

    # Adding axes annotations
    plt.xlabel('Uso de CPU [%]')
    plt.ylabel('Uso de RUN QUEUE [q]')
    plt.title('Separação das classes - SVM')
    plt.show()

    print("GERANDO MATRIZ DE CONFUSAO")

    start = time()
    svc_y_pred = svm_classifier.predict(X_test)
    print(f"DURACAO PREDICAO: {time() - start}")

    print("Matriz de confusao")
    print(confusion_matrix(y_test, svc_y_pred))

    print("---------------------------------")
    print("Cross tab")
    print(pd.crosstab(y_test, svc_y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    print("---------------------------------")
    print("Diagnostico de classificacao")
    print(classification_report(y_test,svc_y_pred)) 
    

if __name__ == "__main__":
    main()