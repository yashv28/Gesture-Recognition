README
-------

Place the test data in 'test_data' folder.

1. Run hmm.py to calculate the predicted gesture for test data files.

	After clustering the continuous data with k-means training data is saved as --> 'discrete_data.npy'.

	After training the different gestures using Baum-Welch algorithm the lambda parameters (pi, A, B) and clustered centoids are saved as --> 'hmm_parameters.npy'.

	NOTE - All variables to toggle training, changing values of N and M are in 'hmm.py' at beginning.

2. hmm_helper.py contains neccessarry functions for forward-backward algorithm and feature transformation.
