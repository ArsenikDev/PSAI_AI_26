import numpy as np
import matplotlib.pyplot as plt
import itertools

n = 8
E_e = 0.01
alpha_fixed = 0.1
np.random.seed(42)

X_full = np.array(list(itertools.product([0, 1], repeat=n)))
y_full = np.array([0 if np.all(x == 1) else 1 for x in X_full])

idx_0 = np.where(y_full == 0)[0]
idx_1 = np.where(y_full == 1)[0]

np.random.seed(42)
np.random.shuffle(idx_0)
np.random.shuffle(idx_1)

train_0_len = 1
train_1_len = int(0.8 * len(X_full)) - 1

train_idx = np.concatenate((idx_0[:train_0_len], idx_1[:train_1_len]))
test_idx = idx_1[train_1_len:]

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_test, y_test = X_full[test_idx], y_full[test_idx]

X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]
X_full_b = np.c_[np.ones(len(X_full)), X_full]


def sigmoid(net):
    return 1 / (1 + np.exp(-np.clip(net, -100, 100)))


def train_perceptron(X, y, loss_func, step_mode):
    np.random.seed(42)
    w = np.random.uniform(-0.1, 0.1, n + 1)
    errors_history = []

    for epoch in range(10000):
        total_error = 0
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]

            y_pred = sigmoid(np.dot(w, xi))

            if step_mode == 'fixed':
                alpha = alpha_fixed
            else:
                alpha = 1.0 / (1.0 + np.sum(xi ** 2))

            if loss_func == 'MSE':
                total_error += 0.5 * (yi - y_pred) ** 2
                grad = (yi - y_pred) * y_pred * (1 - y_pred)
            else:
                eps = 1e-15
                y_p_clip = np.clip(y_pred, eps, 1 - eps)
                total_error += - (yi * np.log(y_p_clip) + (1 - yi) * np.log(1 - y_p_clip))
                grad = (yi - y_pred)

            w += alpha * grad * xi

        errors_history.append(total_error)
        if total_error <= E_e:
            break

    return w, errors_history, epoch + 1


configs = {
    'MSE + Fixed': ('MSE', 'fixed'),
    'MSE + Adaptive': ('MSE', 'adaptive'),
    'BCE + Fixed': ('BCE', 'fixed'),
    'BCE + Adaptive': ('BCE', 'adaptive')
}

print("Обучение запущено. Подождите ")
results = {}
for name, (l_func, s_mode) in configs.items():
    results[name] = train_perceptron(X_train_b, y_train, l_func, s_mode)
    print(f"{name} завершено. Эпох: {results[name][2]}")

plt.figure(figsize=(10, 6))
for name, (w, errs, epochs) in results.items():
    plt.plot(errs, label=f"{name} ({epochs} ep)")

plt.axhline(y=E_e, color='r', linestyle='--', label=f'Порог Ee = {E_e}')
plt.yscale('log')
plt.title('График сходимости: MSE vs BCE')
plt.xlabel('Эпохи')
plt.ylabel('Суммарная ошибка (логарифмическая шкала)')
plt.legend()
plt.grid(True)
plt.show()


def get_accuracy(w, X, y):
    preds = (sigmoid(np.dot(X, w)) >= 0.5).astype(int)
    return np.mean(preds == y) * 100


print("\nРезультаты ")
for name, (w, errs, epochs) in results.items():
    acc_tr = get_accuracy(w, X_train_b, y_train)
    acc_te = get_accuracy(w, X_test_b, y_test)
    acc_f = get_accuracy(w, X_full_b, y_full)
    print(f"{name:<15} | Ep: {epochs:<5} | Train: {acc_tr:>5.1f}% | Test: {acc_te:>5.1f}% | Full: {acc_f:>5.1f}%")

print("\nРежим функционирования")
w_final = results['BCE + Adaptive'][0]
try:
    user_input = input("Введите вектор (8 бит через пробел): ")
    x_val = np.array([int(i) for i in user_input.split()])
    x_val_b = np.insert(x_val, 0, 1)
    prob = sigmoid(np.dot(w_final, x_val_b))
    pred_class = 1 if prob >= 0.5 else 0
    true_class = 0 if np.all(x_val == 1) else 1
    match = "Совпадает с таблицей истинности" if pred_class == true_class else "Расхождение"
    print(f"Вероятность ŷ: {prob:.4f}")
    print(f"Предсказанный класс: {pred_class}")
    print(match)
except Exception:
    pass