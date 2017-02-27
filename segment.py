import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import sklearn
from librosa import load,stft,logamplitude
import cmath
import math
import scipy
import random
#TODO implement different method of obtaining feature vectors
#initialize globals
hop_length = 1024
n_fft = 2048

def load_music(path):
    """
    Returns: loaded music and sample rate given a string of a path to the audio file
    path: string
    """
    music, sr = load(path)
    return music,sr

def feature_vectors(music,hop_length,window_size):
    """
    Returns: feature vectors for audio sample music from the log spectrogram of the music
    music: 1-D numpy array
    hop_length: int
    window_size: int
    """
    _stft = stft(music, hop_length = hop_length, n_fft = n_fft)
    log_spectrogram = logamplitude(np.abs(_stft**2), ref_power=np.max)
    return log_spectrogram

def sim_matrix(feature_vectors, sample_rate, hop_length, distance_metric = 'cosine',display=True):
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
    sim_mat = 1 - dist_mat/dist_mat.max()
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
    return sim_mat

def gaussian_checkerboard_kernel(M):
    """
    Returns: 2-D gaussian checkerboard kernel of size M (even)
    M: int
    """
    g = scipy.signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape((-1, 1)), g.reshape((1, -1)))
    G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
    G[:M / 2, M / 2:] = -G[:M / 2, M / 2:]
    return G

def novelty_curve(sim_mat,ker_size):
    """
    Returns: 1-D numpy array of the novelty curve of similarity matrix sim_mat and kernel size ker_size
    sim_mat: 2-D numpy array of similarity matrix
    ker_size: int of kernel size
    """
    #compute similarity matrix in lag domain
    # r = np.floor(ker_size/2)
    # s_lag = np.zeros((sim_mat.shape[0],sim_mat.shape[1] + ker_size - r))
    # for i in range(sim_mat.shape[0]):
    #     for l in range(ker_size):
    #         s_lag[i,l] = sim_mat[i,i+l-r]
    novelty_curve = np.zeros(sim_mat.shape[0])
    padded_sim_mat = np.pad(sim_mat,ker_size/2,'constant')

    kernel = gaussian_checkerboard_kernel(ker_size)
    i = 0
    for r in range(ker_size/2,sim_mat.shape[0]+ker_size/2):
        novelty_curve[r-ker_size/2] = np.sum(padded_sim_mat[r-ker_size/2:r+ker_size/2,r-ker_size/2:r+ker_size/2]*kernel)

    #normalize
    novelty_curve += novelty_curve.min()
    novelty_curve = novelty_curve/novelty_curve.max()
    return novelty_curve

hop_length = 1024
window_size = 2048
music,sr = load_music("./call_me_maybe.mp3")
feature_vectors = feature_vectors(music,hop_length,window_size)
sim_mat = sim_matrix(feature_vectors,sr,hop_length,display=False)
novelty_curve = novelty_curve(sim_mat,256)

plt.figure(0)
plt.plot(novelty_curve)

plt.figure(1)
plt.imshow(sim_mat)
plt.show()
