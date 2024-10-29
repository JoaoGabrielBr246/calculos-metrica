import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dados de exemplo
# Verdadeiros (rótulos reais)
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# Previsões (rótulos previstos pelo modelo)
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

# Calcula a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)

# Extrai os valores de TN, FP, FN, TP
TN, FP, FN, TP = conf_matrix.ravel()

# Funções para calcular as métricas
def calcular_acuracia(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def calcular_sensibilidade(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def calcular_especificidade(TN, FP):
    return TN / (TN + FP) if (TN + FP) != 0 else 0

def calcular_precisao(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def calcular_fscore(precisao, sensibilidade):
    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) != 0 else 0

# Cálculo das métricas
acuracia = calcular_acuracia(TP, TN, FP, FN)
sensibilidade = calcular_sensibilidade(TP, FN)
especificidade = calcular_especificidade(TN, FP)
precisao = calcular_precisao(TP, FP)
fscore = calcular_fscore(precisao, sensibilidade)

# Exibindo os resultados
print(f"Acurácia: {acuracia:.2f}")
print(f"Sensibilidade (Recall): {sensibilidade:.2f}")
print(f"Especificidade: {especificidade:.2f}")
print(f"Precisão: {precisao:.2f}")
print(f"F-score: {fscore:.2f}")

# Exibe a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Classe 0", "Classe 1"])
disp.plot(cmap=plt.cm.Blues)
plt.show()
