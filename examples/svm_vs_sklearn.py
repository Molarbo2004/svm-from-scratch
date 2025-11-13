import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from manually_svm import ManuallySVM
from sklearn import svm
from sklearn.metrics import accuracy_score

# Генерируем нелинейные данные
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
y = np.where(y == 0, -1, 1)  # Преобразуем в {-1, 1}

print("NON-LINEAR DATASET: Make Moons")
print(f"Количество примеров: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Параметры для сравнения
C = 100  # sklearn использует C (1/alpha)
alpha = 1/C  # Наш параметр регуляризации

print(f"\nПараметры сравнения:")
print(f"Sklearn C = {C}")
print(f"Наш альфа = {alpha}")

# Наша реализация (линейное ядро)
model_custom = ManuallySVM(lr=0.001, alpha=alpha, epochs=1000)
model_custom.fit(X_train, y_train)

# Sklearn реализация (линейное ядро)
model_sklearn = svm.SVC(kernel='linear', C=C)
model_sklearn.fit(X_train, y_train)

# Оценка нашей модели
train_acc_custom = model_custom.score(X_train, y_train)
test_acc_custom = model_custom.score(X_test, y_test)
train_margin_acc_custom = model_custom.margin_score(X_train, y_train)
test_margin_acc_custom = model_custom.margin_score(X_test, y_test)

# Оценка sklearn модели
train_acc_sklearn = accuracy_score(y_train, model_sklearn.predict(X_train))
test_acc_sklearn = accuracy_score(y_test, model_sklearn.predict(X_test))

print(f"\n{'='*60}")
print(f"СРАВНЕНИЕ РЕЗУЛЬТАТОВ НА НЕЛИНЕЙНЫХ ДАННЫХ")
print(f"{'='*60}")

print(f"\nНАША РЕАЛИЗАЦИЯ (Линейное ядро):")
print(f"Train Accuracy (только знак): {train_acc_custom:.4f}")
print(f"Test Accuracy (только знак): {test_acc_custom:.4f}")
print(f"Train Accuracy (с margin): {train_margin_acc_custom:.4f}")
print(f"Test Accuracy (с margin): {test_margin_acc_custom:.4f}")

print(f"\nSKLEARN Линейное ядро:")
print(f"Train Accuracy: {train_acc_sklearn:.4f}")
print(f"Test Accuracy: {test_acc_sklearn:.4f}")

print(f"\nПАРАМЕТРЫ МОДЕЛЕЙ:")
print(f"Наши веса: [{model_custom.weights[0]:.6f}, {model_custom.weights[1]:.6f}]")
print(f"Наш bias: {model_custom.bias:.6f}")
print(f"Sklearn веса: [{model_sklearn.coef_[0][0]:.6f}, {model_sklearn.coef_[0][1]:.6f}]")
print(f"Sklearn bias: {model_sklearn.intercept_[0]:.6f}")

print(f"\nРАЗНИЦА:")
print(f"Train Accuracy разница: {abs(train_acc_custom - train_acc_sklearn):.4f}")
print(f"Test Accuracy разница: {abs(test_acc_custom - test_acc_sklearn):.4f}")

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Подготовка данных для визуализации
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# График 1: Наша реализация
Z_custom = model_custom.predict(np.c_[xx.ravel(), yy.ravel()])
Z_custom = Z_custom.reshape(xx.shape)
ax1.contourf(xx, yy, Z_custom, alpha=0.8, cmap=plt.cm.RdBu)
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=50)

# Линии margin для нашей реализации
w0, w1 = model_custom.weights
w2 = model_custom.bias
x_line = np.linspace(x_min, x_max, 100)
if abs(w1) > 1e-6:  # избегаем деления на 0
    y_main = (-w0 * x_line - w2) / w1
    y_upper = (-w0 * x_line - w2 + 1) / w1
    y_lower = (-w0 * x_line - w2 - 1) / w1
    ax1.plot(x_line, y_main, 'black', linewidth=2, label='Разделяющая линия')
    ax1.plot(x_line, y_upper, 'red', '--', linewidth=1, alpha=0.7)
    ax1.plot(x_line, y_lower, 'blue', '--', linewidth=1, alpha=0.7)

ax1.set_xlabel('Признак 1')
ax1.set_ylabel('Признак 2')
ax1.set_title(f'НАША РЕАЛИЗАЦИЯ\n Тестовая Точность: {test_acc_custom:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Sklearn Linear
Z_sklearn = model_sklearn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_sklearn = Z_sklearn.reshape(xx.shape)
ax2.contourf(xx, yy, Z_sklearn, alpha=0.8, cmap=plt.cm.RdBu)
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=50)

# Линии margin для sklearn
w0_sk, w1_sk = model_sklearn.coef_[0]
w2_sk = model_sklearn.intercept_[0]
if abs(w1_sk) > 1e-6:
    y_main_sk = (-w0_sk * x_line - w2_sk) / w1_sk
    y_upper_sk = (-w0_sk * x_line - w2_sk + 1) / w1_sk
    y_lower_sk = (-w0_sk * x_line - w2_sk - 1) / w1_sk
    ax2.plot(x_line, y_main_sk, 'black', linewidth=2, label='Разделяющая линия')
    ax2.plot(x_line, y_upper_sk, 'red', '--', linewidth=1, alpha=0.7)
    ax2.plot(x_line, y_lower_sk, 'blue', '--', linewidth=1, alpha=0.7)

ax2.set_xlabel('Признак 1')
ax2.set_ylabel('Признак 2')
ax2.set_title(f'SKLEARN LINEAR\nТестовая точность: {test_acc_sklearn:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()