import numpy as np

def forward_backward(features, A, B, pi):

	T = features.shape[0]
	N = A.shape[0]

	coeff = np.zeros((T, 1))

	alpha = np.zeros((T, N))
	alpha[0] = pi[:, 0] * B[features[0]]
	coeff[0] = 1. / np.maximum(1E-10, np.sum(alpha[0]))
	alpha[0] *= coeff[0]

	# Forward
	for t in range(0, T-1):
		alpha[t+1] = np.sum(alpha[t][:, np.newaxis] * A, axis=0) * B[features[t+1]]
		coeff[t+1] = 1. / np.maximum(1E-10, np.sum(alpha[t+1]))
		alpha[t+1] *= coeff[t+1]
		alpha[t+1] = np.clip(alpha[t+1], a_min=1E-100, a_max=1)

	P_0 = -np.sum(np.log(coeff))

	beta = np.zeros((T, N))
	beta[-1] = 1
	beta[-1] *= coeff[-1]

	# Backward
	for t in range(T-2, -1, -1):
		beta[t] = np.sum(A * B[features[t+1]][np.newaxis, :] * beta[t+1][np.newaxis, :], axis=1)
		beta[t] *= coeff[t]

	return alpha, beta, P_0, coeff


def txt2features(txt):
	T = txt.shape[0]
	features = np.zeros((T, 7))
	features[:, 0] = np.linalg.norm(txt[:, [0, 1, 2]], axis=1)
	features[:, 1] = np.arctan2(txt[:, 1], txt[:, 0])
	features[:, [2, 3, 4]] = txt[:, [0, 1, 2]]
	features[:, [5, 6]] = txt[:, [3, 4]]

	return features
