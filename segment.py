import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import librosa
import cmath
import math
import scipy
import random

#import audio
music, sr = librosa.load('./call_me_maybe.mp3')
hop_length = 1024
n_fft = 2048


stft = librosa.stft(music, hop_length = hop_length, n_fft = n_fft)
log_spectrogram = librosa.logamplitude(np.abs(stft**2), ref_power=np.max)

feature_vectors = log_spectrogram #for simplified naming
def sim_matrix(feature_vectors, sample_rate, hop_length, distance_metric = 'cosine', display = True):
    """
        Input:
            feature_vectors - a numpy ndarray MxN, where M is the number of features in each vector and
            N is the length of the sequence.
            sample_rate - sample rate of the original audio
            hop_length - the length of the hop used in the representation
            distance_metric - which distance metric to use to compute similarity. Defaults to cosine.
            display - whether or not to display the similarity matrix after computing it. Defaults to True.
        Output:
            if display is True, plot the similarity matrix. Along the x and y axis of the similarity matrix,
            the ticks should be in seconds not in samples.
            returns sim_matrix - an NxN matrix with the pairwise distance between every feature vector.
    """
    dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(feature_vectors.T,distance_metric))
    sim_mat = 1 - dist_mat/np.max(dist_mat)

    if display:
        plt.figure()
        plt.imshow(sim_mat)
        skip = feature_vectors.shape[-1] / 10
        plt.xticks(np.arange(0, feature_vectors.shape[-1], skip),
                               ['%.2f' % (i * hop_length / float(sample_rate)) for i in range(feature_vectors.shape[-1])][::skip],
                  rotation='vertical')
        plt.yticks(np.arange(0, feature_vectors.shape[-1], skip),
                               ['%.2f' % (i * hop_length / float(sample_rate)) for i in range(feature_vectors.shape[-1])][::skip])
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')
        plt.title('Similarity matrix')
        plt.show()
    return sim_mat

def gaussian_checkerboard_kernel(M):
    """ Creates gaussian checkerboard kernel of size M (even)"""
    g = scipy.signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape((-1, 1)), g.reshape((1, -1)))
    G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
    #G[:M / 2, M / 2:] = -G[:M / 2, M / 2:]
    return G
