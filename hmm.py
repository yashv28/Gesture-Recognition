from sklearn.cluster import KMeans
import os
from hmm_helper import *

#######################################################################################
#######################################################################################

trainfolder = 'train_data'
testfolder = 'test_data'

N = 10
M = 30
max_iters = 1000
diff = 0.001

train = False #Set True for training

#######################################################################################
#######################################################################################

#K-Means Clustering

if not os.path.exists('discrete_data.npy'):
	data = {}
	gestures = []
	
	for filename in os.listdir(trainfolder):

		file = filename.split('.')[0]
		gesture = file.split('_')[0]
		txtfile = np.loadtxt(os.path.join(trainfolder, filename))

		features = txt2features(txtfile[:, 1:])
		data[file] = {'raw': features}

		if gesture not in gestures:
			gestures.append(gesture)

	data['gestures'] = gestures

	gesture_data = [(key, value['raw']) for key, value in data.items() if 'gestures' not in key.lower()]
	for idx, set in enumerate(gesture_data):
		stacked_data = set[1] if idx == 0 \
			else np.vstack((stacked_data, set[1]))

	# KMeans centroids
	kmeans = KMeans(n_clusters=M, init='k-means++').fit(stacked_data)
	centroids = kmeans.cluster_centers_
	data['centroids'] = centroids

	for set in gesture_data:
		file = set[0]
		predicted_labels = kmeans.predict(set[1])
		data[file]['discrete'] = predicted_labels

	np.save('discrete_data.npy', data)

	TrainData = np.load('discrete_data.npy').item()

else:
	TrainData = np.load('discrete_data.npy').item()

#######################################################################################
#######################################################################################

#Training

if train:
	for gesture in TrainData['gestures']:

		print '\n{0} model'.format(gesture)
		print '------------------------------------------------------------'

		pi = np.zeros((N, 1))
		pi[0] = 1

		A = np.random.uniform(low=0.05, high=1, size=(N, N))
		A = np.triu(np.sort(A)[:, ::-1])
		A /= np.sum(A, axis=1)[:, np.newaxis]

		B = np.random.uniform(low=0.1, high=1, size=(M, N))
		B /= np.sum(B, axis=0)[np.newaxis, :]

		# Baum-Welch (EM)
		gesture_data = [value['discrete'] for key, value in TrainData.items() if gesture in key.lower()]

		c = 0
		while c < max_iters:

			c += 1

			xi_sum = np.zeros((N, N))
			gamma_sum = np.zeros((1, N))
			feat_count = np.zeros((M, N))
			likelihood = np.zeros(len(gesture_data))

			for idx, features in enumerate(gesture_data):

				# E-step
				alpha, beta, P_0, coeff = forward_backward(features, A, B, pi)
				likelihood[idx] = P_0

				T, N = alpha.shape
				xi = np.zeros((N, N, T-1))
				for t in range(T-1):
					xi[:, :, t] = alpha[t][:, np.newaxis] * A * B[int(features[t + 1])][np.newaxis, :] * beta[t + 1][np.newaxis, :]

				xi_sum += np.sum(xi, axis=2)

				gamma = np.zeros((T, N))
				for t in range(T):
					gamma[t] = alpha[t] * beta[t] * (1. / coeff[t])
				gamma_sum += np.sum(gamma, axis=0)

				for l in range(M):
					feat_i = (features == l)
					feat_count[l] += np.sum(gamma[feat_i], axis=0)

			newmeanll = np.mean(likelihood)
			print '{0} -> Iteration: {1},  Mean Likelihood: {2}'.format(gesture,c,newmeanll)

			# M-step
			newA = xi_sum / gamma_sum
			newA /= np.sum(newA, axis=1)[:, np.newaxis]
			newB = feat_count / gamma_sum
			newB /= np.sum(newB, axis=0)[np.newaxis, :]

			if c > 1 and abs(newmeanll - meanll) < diff: break
			
			A = newA
			B = newB

			meanll = np.mean(likelihood)

		TrainData[gesture + '_lambda'] = {'pi': pi, 'A': A, 'B': B}

	np.save('hmm_parameters.npy', TrainData)

	TrainData = np.load('hmm_parameters.npy').item()

else:
	TrainData = np.load('hmm_parameters.npy').item()


######################################################################################
######################################################################################

acc = []
print '\nPrediction (Please place test data in \'test_data\' folder)\n'
for filename in os.listdir(testfolder):

	txtfile = np.loadtxt(os.path.join(testfolder, filename))
	file = filename.split('.')[0]

	features = txt2features(txtfile[:, 1:])

	likelihoods = []
	for gesture in TrainData['gestures']:

		centroids = TrainData['centroids']
		k = centroids.shape[0]
		labels = KMeans(n_clusters=k, init=centroids, n_init=1).fit(features).labels_

		pi = TrainData[gesture + '_lambda']['pi']
		A = TrainData[gesture + '_lambda']['A']
		B = TrainData[gesture + '_lambda']['B']

		_, _, P_0, _ = forward_backward(labels, A, B, pi)
		likelihoods.append(P_0)

	prediction = TrainData['gestures'][np.argmax(np.array(likelihoods))]
	print np.array(likelihoods)
	maxll = np.amax(np.array(likelihoods))
	print 'File: {0}, Prediction: {1}, Max Likelihood: {2}\n'.format(file,prediction,maxll)
	acc.append(prediction in file)

accuracy = 100*float(np.sum(acc)) / len(acc)
print '\nAccuracy: {0}%'.format(accuracy)
