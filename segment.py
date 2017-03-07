import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import sklearn
import librosa
import cmath
import math
import scipy
import random
import peakdetect
import time
#TODO implement different method of obtaining feature vectors


def load_music(path):
    """
    Returns: loaded music and sample rate given a string of a path to the audio file
    path: string
    """
    music, sr = librosa.load(path)
    return music,sr

def feature_vectors(music,sample_rate,hop_length,window_size,method = 'stft',beat_sync = True):
    """
    Returns: feature vectors for audio sample music from the log spectrogram of the music
    music: 1-D numpy array
    sample_rate: sample rate
    hop_length: int
    window_size: int
    method: string, method for computing feature vectors Default to stft. ('stft','mfcc','cqt','tempogram')
    """
    if method == 'stft':
        #feature_vectors = librosa.feature.chroma_stft(y=music,sr=sample_rate)
        #feature_vectors = librosa.logamplitude(np.abs(_stft**2), ref_power=np.max)
        _stft = librosa.stft(music, hop_length = hop_length, n_fft = 2048)
        feature_vectors = librosa.logamplitude(np.abs(_stft**2), ref_power=np.max)
    elif method == 'cqt':
        feature_vectors = librosa.feature.chroma_cqt(y=music,sr=sample_rate)
    elif method == 'mfcc':
        feature_vectors = librosa.feature.mfcc(y=music, sr=sample_rate)
    elif method == 'tempogram':
        feature_vectors = librosa.feature.tempogram(y=music,sr=sample_rate)
    if beat_sync:
        feature_vectors = beat_sync_features(music,sample_rate,hop_length,feature_vectors,aggregator = np.median)
    return feature_vectors

def sim_matrix(feature_vectors, sample_rate, hop_length, distance_metric = 'euclidean',display=True):
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

def compute_novelty_curve(sim_mat,ker_size):
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

def pick_peaks(nc,):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    #th = np.mean(nc)/2
    offset = nc.mean() / 3
    th = scipy.ndimage.filters.median_filter(nc, size=32) + offset
    peaks = []
    for i in xrange(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    return peaks,th

def beat_sync_features(music,sample_rate,hop_length,feature_vectors,aggregator = np.median):
    """
        input:
            feature_vectors: a numpy ndarray MxN, where M is the number of features in each vector and
            N is the length of the sequence.
            beats: frames given by the beat tracker
            aggregator: how to summarize all the frames within a beat (e.g. np.median, np.mean). Defaults to np.median.
            display: if True, displays the beat synchronous features.
        output:
            beat_synced_features: a numpy ndarray MxB, where M is the number of features in each vector
            and B is the number of beats. Each column of this matrix represents a beat synchronous feature
            vector.
    """
    tempo,beats = librosa.beat.beat_track(y=music, sr=sr, hop_length=hop_length)
    bsf = np.zeros((feature_vectors.shape[0],beats.size))
    for b in range(beats.size):
        if b ==0:
            temp = feature_vectors[:,0:beats[0]]
        elif b==beats.size-1:
            temp = feature_vectors[:,beats[b]:]
        else:
            temp = feature_vectors[:,beats[b-1]:beats[b]]
        for i in range(temp.shape[0]):
            bsf[i,b] = aggregator(temp[i,:])
    return bsf

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.filter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def segment_cluster(sim_mat,bounds):
    """
    Returns:
    sim_mat: 2-D numpy array of similarity matrix
    bounds: indices of the boundaries indicating segment boundaries
    """
    segments = []
    boundaries = np.append(boundaries,0,1)
    if boundaries[0] == 0:
        print "GOOD"
    for i in range(1,boundaries.size):
        segments[i] = sim_mat[boundaries[i-1]:boundaries[i],boundaries[i-1]:boundaries[i]]
        _,eigen,_ = numpy.linalg.svd(segments[i],full_matrices=0,compute_uv=0)



##LOAD MUSIC, GET FEATURES, SIM MATRIX
hop_length = 512
window_size = 2048
ker_size = 64
smoothing_window = 4
start = time.time()
music,sr = load_music("./Empire.mp3")
end=time.time()
print "Loading took %d seconds" % (end-start)

start = time.time()
feature_vectors = feature_vectors(music,sr,hop_length,window_size,method = 'stft')
end = time.time()

print "feature_vectors took %d seconds" % (end-start)

start = time.time()
sim_mat = sim_matrix(feature_vectors,sr,hop_length,distance_metric = 'euclidean',display=False)
print sim_mat.shape
end=time.time()

print "sim_mat took %d seconds" % (end-start)
#GET NOVELTY CURVE OF SIMILARITY MATRIX

start = time.time()
novelty_curve = compute_novelty_curve(sim_mat,ker_size)
end=time.time()

print "novelty took %d seconds" % (end-start)
#novelty_curve = novelty_curve[100:8000] #the end is wonky

start=time.time()
novelty_curve_smooth = smooth(novelty_curve,window_len = smoothing_window)
end=time.time()

print "smoothing took %d seconds" % (end-start)
#RECURRENCE MATRIX
#recurr = librosa.segment.recurrence_matrix(feature_vectors,mode='affinity')
#novelty_curve_recurr = compute_novelty_curve(recurr,ker_size/2)
#novelty_curve_recurr = novelty_curve_recurr[100:8000] #the end is wonky
#novelty_curve_recurr_smooth = smooth(novelty_curve_recurr,window_len = smoothing_window)


#CALCULATE DERIVATIVE
# deriv = np.zeros(novelty_curve_smooth.size)
# for i in range(1,novelty_curve_smooth.size - 1):
#     deriv[i-1] = (novelty_curve_smooth[i] - novelty_curve_smooth[i-1])
# magnify = 10
# deriv = magnify*deriv

#PICK PEAKS
start = time.time()
peaks,th= pick_peaks(novelty_curve)
end = time.time()

print "peak picking took %d seconds" % (end-start)
# peaks_recurr = pick_peaks(novelty_curve_recurr_smooth)

#PRINT DERIVATIVE VALUES
# deriv_pts = [deriv[i] for i in peaks]
# print deriv_pts


tempo,beats = librosa.beat.beat_track(y=music,sr=sr,hop_length=hop_length)

beat_times = np.zeros(beats.size)
for i in range(beats.size):
    beat_times[i] = (beats[i]*hop_length)/float(sr)

#PLOTTING
start = time.time()
skip = beats.shape[-1] / 10
plt.figure(1)
plt.title('STFT')
plt.plot(novelty_curve,color='g')
#plt.plot(novelty_curve_smooth,color='r')
plt.plot(th)
#plt.axhline(y=np.std(novelty_curve_smooth)/2,color='orange')
#plt.xticks(np.arange(0, beats.shape[-1]), ['%.2f' % (i * hop_length / float(sr)) for i in range(beats.shape[-1])][::skip])

#plt.plot(deriv,color='blue')
for p in peaks:
   plt.axvline(x=p,color='m')

plt.figure(2)
plt.title('STFT')
#plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
plt.imshow(sim_mat)
end = time.time()

print "plotting took %d seconds" % (end-start)
start = time.time()
# plt.figure(3)
# plt.title('RECURR')
# plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# plt.imshow(recurr)
#
# plt.figure(4)
# plt.title('RECURR')
# plt.plot(novelty_curve_recurr,color='g')
# plt.plot(novelty_curve_recurr_smooth,color='r')
# plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# for p in peaks_recurr:
#     plt.axvline(x=p,color='r')
plt.show()
end = time.time()

print "showing took %d seconds" % (end-start)
