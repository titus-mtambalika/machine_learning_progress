import numpy as np
from math import e
from collections import Counter

def track_loss(model, X, y, epochs = 1000, i = 50, as_percentage = False, learning_rate = 0.01):
	loss_report = np.array([])
	# epochs for total iterations i for epochs per iterations
	if not as_percentage:
		for i in range(epochs // i):
			model.fit(X, y, epochs = i, learning_rate = learning_rate)
			loss = np.sum((y - model.predict(X)) ** 2) / X.shape[0]
			loss_report = np.append(loss_report, loss)
	else:
		for i in range(epochs // i):
			model.fit(X, y, epochs = i, llearning_rate = learning_rate)
			loss = np.mean(((y - model.predict(X)) ** 2) / y) * 100
			loss_report = np.append(loss_report, loss)
	return loss_report

def regularize(X, axis = 0):
	reg = X - np.min(X, axis = axis)
	return reg / np.mean(reg, axis = 0)

def sigmoid(z):
	return 1 / (1 + e ** -z)
	
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
		
	def predic(self, samples, K = 7):
		return [self.predict_one(sample) for sample in samples]
		
class GaussianNB:
	def __init__(self, epsilon = 1e-9):
		self.classes = None
		self.epsilon = epsilon
		self.mean