import numpy as np
import math
from collections import Counter

def track_loss(model, X, y, epochs = 1000, i = 50, as_percentage = False, learning_rate = 0.01):
	loss_report = np.array([])
	# epochs for total iterations i for epochs per iterations
	if not as_percentage:
		for i in range(epochs // i):
			model.fit(X, y, epochs = i, learning_rate = learning_rate)
			loss = np.sum((y - model.predict(X)) ** 2) / y.shape[0]
			loss_report = np.append(loss_report, loss)
	else:
		for i in range(epochs // i):
			model.fit(X, y, epochs = i, learning_rate = learning_rate)
			loss = np.mean(((y - model.predict(X)) ** 2) / y) * 100
			loss_report = np.append(loss_report, loss)
	return loss_report

def regularize(X, axis = 0):
	reg = X - np.min(X, axis = axis)
	return reg / np.mean(reg, axis = 0)

def sigmoid(z):
	return 1 / (1 + math.e ** -z)
	
def asigmoid(z):
	return -1 * np.log((1 / z) - 1)

class Lin_reg:
	def __init__(self):
		self.weights = None
		self.bias = None
	
	def fit(self, X, y, epochs = 3000, learning_rate = 0.01):
		n_samples, n_features = X.shape
		self.loss_report = np.array([])
		if self.weights is None:
			self.weights = np.zeros((n_features, 1))
		if self.bias is None:
			self.bias = 0 
			
		for epoch in range(epochs):
			prediction = X @ self.weights + self.bias
			loss = np.sum((prediction - y) ** 2)
			
			self.weights -= (learning_rate / n_samples) * (X.T @ (prediction - y))
			self.bias -= learning_rate * np.mean(prediction - y)
			
	def predict(self, X):
		return X @ self.weights + self.bias
		
class Log_reg:
	def __init__(self):
		self.weights = None
		self.bias = None
		
	def fit(self, X, y, epochs = 50000, learning_rate=0.01):
		n_samples, n_features = X.shape
		self.weights = np.zeros((n_features, 1))
		self.bias = 0
		
		for epoch in range(epochs):
			z = X @ self.weights + self.bias
			y_prediction = sigmoid(z)
			
			# Gradients (simplified form)
			dl_dweights = X.T @ (y_prediction - y)
			dl_dbias = np.sum(y_prediction - y)
			
			# Update parameters
			self.weights -= learning_rate * dl_dweights
			self.bias -= learning_rate * dl_dbias
	
	def predict_proba(self, X):
		z = X @ self.weights + self.bias
		return sigmoid(z) 
	
	def predict(self, X, threshold=0.5):
		proba = self.predict_proba(X)
		return (proba >= threshold).astype(int)
	
	
class KNN:
	def __init__(self, X, y):
		self.X = X 
		self.y = y
		
	def classify_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		votes = nearest_neighbours[:K]
		return Counter(votes).most_common()[0][0]
		
	def classify(self, samples, K = 7):
		return [self.classify_one(sample) for sample in samples]
		
	
	def predict_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		votes = nearest_neighbours[:K]
		return np.mean(votes, axis = 1)
		
	def predict(self, samples, K = 7):
		return [self.predict_one(sample) for sample in samples]
		
class GaussianNB:
	def __init__(self, epsilon=1e-9):
		self.epsilon = epsilon
		self.classes = None
		self.priors = None	  # dict: class -> prior
		self.means = None	   # dict: class -> mean vector
		self.variances = None   # dict: class -> variance vector
		#cache for faster predictions
		self.bases = None
	
	def fit(self, X, y):
		self.classes = np.unique(y)
		self.priors = {}
		self.means = {}
		self.variances = {}
		self.bases = {}
		

		for c in self.classes:
			#flatten mask to avoid index error
			X_c = X[(y == c).flatten()]
			self.priors[c] = len(X_c) / len(X)
			self.means[c] = np.mean(X_c, axis=0)
			self.variances[c] = np.var(X_c, axis=0) + self.epsilon
			#cache bases for faster _log_likelihood
			self.bases[c] = 1 / (2 * math.pi * self.variances[c])
		
	
	def _log_likelihood(self, x, c):
		# Compute log P(x_j | y=c) for all j
		density = self.bases[c] ** -((x - self.means[c]) ** 2 / (2 * self.variances[c]))
		# Then sum them
		return np.sum(density) + self.priors[c]
	
	def predict(self, X):
		predictions = np.array([None] * X.shape[0])
		for i, sample in enumerate(X):
			predicted_class, score = self.classes[0], 0
			for c in self.classes:
				density = self._log_likelihood(sample, c)
				if density > score:
					predicted_class, score = c, density
			predictions[i] = predicted_class
		return predictions