import numpy as np

class ManuallySVM():


    ### НАЧАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ###


    def __init__(self, lr = 0.01, alpha = 0.1, epochs = 200):
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    # Функция добавления смещения 

    def add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([X, ones])
    
    # Функция обучения

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.Y_train = y_train

        y_train = np.where(y_train <= 0, -1, 1)

        # Добавляем фиктивный признак для bias
        X_extended = self.add_bias(X_train)
        m, n = X_extended.shape

        # Инициализируем веса (включая bias как последний элемент)
        self.weights = np.random.normal(loc=0, scale=0.05, size=n)

        for epoch in range(self.epochs):
            for i in range(m):
                 x_i = X_extended[i]
                 y_i = y_train[i]

                 # Вычисляем отступ (margin): y * (w.T @ x)
                 margin = y_i * np.dot(self.weights, x_i)

                 if margin >= 1:
                     grad = self.alpha * self.weights

                 else:
                     grad = self.alpha * self.weights - y_i * x_i
                
                 # Обновляем веса
                 self.weights -= self.lr * grad

        # Веса включают [w, b], где b = self.weights[-1]
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]  # оставляем только w

    # Функция предсказания 

    def predict(self, X):
        # y_pred = sign(w.T @ x + b)
        y_pred = np.dot(X, self.weights) + self.bias
        return np.sign(y_pred)
    
    # Функция точности

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.where(y <= 0, -1, 1)  # Приводим к -1, +1
        return np.mean(y_pred == y_true)

    def margin_score(self, X, y):

        scores = np.dot(X, self.weights) + self.bias
        y_true = np.where(y <= 0, -1, 1)
        
        # Точка считается правильно классифицированной ТОЛЬКО если:
        # - Для класса +1: score >= 1
        # - Для класса -1: score <= -1
        correct = ((y_true == 1) & (scores >= 1)) | ((y_true == -1) & (scores <= -1))
        
        return np.mean(correct)
                    


                

                 




        
        



