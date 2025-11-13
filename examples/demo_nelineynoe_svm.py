import numpy as np
import sys
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from manually_svm import ManuallySVM

# Генерация данных
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# ПРАВИЛЬНОЕ разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = ManuallySVM(lr=0.001, alpha=0.01, epochs=1000)
model.fit(X_train, y_train)

# Оценка
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
train_margin_acc = model.margin_score(X_train, y_train)
test_margin_acc = model.margin_score(X_test, y_test)

print(f"\nРезультаты:")
print(f"Train Accuracy (только знак): {train_acc:.4f}")
print(f"Test Accuracy (только знак): {test_acc:.4f}")
print(f"Train Accuracy (с margin): {train_margin_acc:.4f}")
print(f"Test Accuracy (с margin): {test_margin_acc:.4f}")

# Визуализация
plt.figure(figsize=(12, 8))
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=60)

# Добавляем линии margin
w0, w1 = model.weights
w2 = model.bias

# Основная разделяющая линия
x_line = np.linspace(x_min, x_max, 100)
y_main = (-w0 * x_line - w2) / w1
y_upper = (-w0 * x_line - w2 + 1) / w1  # Margin +1
y_lower = (-w0 * x_line - w2 - 1) / w1  # Margin -1

plt.plot(x_line, y_main, 'black', linewidth=2, label='Разделяющая линия')
plt.plot(x_line, y_upper, 'red', '--', linewidth=1.5)
plt.plot(x_line, y_lower, 'blue', '--', linewidth=1.5)

plt.colorbar(scatter)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'SVM на Make Moons\nAccuracy (с margin): {test_margin_acc:.3f}, Accuracy (без учета margin): {test_acc:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()