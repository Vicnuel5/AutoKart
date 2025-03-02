import numpy as np

# Función de activación sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función 
def derivative(x):
    return x * (1 - x)

def forward_propagation(X, thetas):
	activations = [X]  # Almacena las activaciones en cada capa
	prev_activation = X

	# Propagación hacia adelante a través de la capa oculta
	for theta in thetas:
		# Añadir el término de sesgo (1's) a la entrada actual
		prev_activation = np.hstack([np.ones((len(prev_activation), 1)), prev_activation])

		# Calcular la entrada ponderada y aplicar la función de activación
		activation = sigmoid(np.dot(prev_activation, theta.T))

		# Almacenar la activación de esta capa para la siguiente iteración
		activations.append(activation)

		# La salida de esta capa será la entrada de la próxima
		prev_activation = activation

	return activations

def backward_propagation(thetas, X, y, lambda_):
	m = len(y)
	deltas = [None] * len(thetas)
	grad = [None] * len(thetas)

	activations = forward_propagation(X, thetas)

	J = cost(thetas, X, y, lambda_, activations[-1])

	# Calcular el error en la capa de salida
	deltas[-1] = activations[-1] - y

	# Calcular el error de las capas ocultas hacia la capa de entrada
	for i in range(len(thetas) - 1, 0, -1):
		# Calcular el delta para la capa actual
		weighted_delta = np.dot(deltas[i], thetas[i][:, 1:])

		# Calcular la derivada sigmoidal para la capa anterior
		activation_sigmoid_derivative = derivative(activations[i])

		# Multiplicar los deltas por la derivada sigmoidal para obtener el nuevo delta
		deltas[i - 1] = weighted_delta * activation_sigmoid_derivative

	# Calcular gradientes con regularización
	for i in range(len(grad)):
		# Calcular el término de regularización L2
		regularization_term = (lambda_ / m) * np.hstack([np.zeros((thetas[i].shape[0], 1)), thetas[i][:, 1:]])

		# Calcular la parte sin regularización del gradiente
		gradient_without_reg = (1 / m) * np.dot(deltas[i].T, np.hstack([np.ones((len(activations[i]), 1)), activations[i]]))

		# Sumar la regularización al gradiente sin regularización
		grad[i] = gradient_without_reg + regularization_term

	return J, grad

def predict(thetas, X):
    return forward_propagation(X, thetas)[-1]

def cost(thetas, X, y, lambda_, yp = None):
	m = X.shape[0]

	if yp is None:
		yp = forward_propagation(X, thetas)[-1]	

	# Cálculo del costo sin regularización
	J = - (1 / m) * np.sum(np.sum(y * np.log(yp[-1]) + (1 - y) * np.log(1 - yp[-1])))

	J += (lambda_ / (2 * m)) * np.sum([np.sum(theta[:, 1:]**2) for theta in thetas])

	return J

def converge(thetas, X, y, lambda_, alpha, iterations):

	J_history = []

	for i in range(iterations):
		J, thetas_grad = backward_propagation(thetas, X, y, lambda_)
		J_history.append(J)
		for i in range(len(thetas)):
			thetas[i] -= alpha * thetas_grad[i]

	return J_history, thetas