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


def load_music(path):
    """
    Returns: loaded music and sample rate given a string of a path to the audio file
    path: string
    """
    music, sr = librosa.load(path)
    return music,sr

def feature_vectors(music,sample_rate,hop_length,window_size,feature_method = 'stft',beat_sync = True):
    """
    Returns: feature vectors for audio sample music from the log spectrogram of the music
    music: 1-D numpy array
    sample_rate: sample rate
    hop_length: int
    window_size: int
    method: string, method for computing feature vectors Default to stft. ('stft','mfcc','cqt','tempogram')
    """
    if feature_method == 'stft':
        #feature_vectors = librosa.feature.chroma_stft(y=music,sr=sample_rate)
        #feature_vectors = librosa.logamplitude(np.abs(_stft**2), ref_power=np.max)
        _stft = librosa.stft(music, hop_length = hop_length, n_fft = 2048)
        feature_vectors = librosa.logamplitude(np.abs(_stft**2), ref_power=np.max)
    elif feature_method == 'cqt':
        feature_vectors = librosa.feature.chroma_cqt(y=music,sr=sample_rate)
    elif feature_method == 'mfcc':
        feature_vectors = librosa.feature.mfcc(y=music, sr=sample_rate)
    elif feature_method == 'tempogram':
        feature_vectors = librosa.feature.tempogram(y=music,sr=sample_rate)
    if beat_sync:
        feature_vectors_beats = beat_sync_features(music,sample_rate,hop_length,feature_vectors,aggregator = np.median)
    return feature_vectors, feature_vectors_beats

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

def pick_peaks(nc,peak_filter_size = 32):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    #th = np.mean(nc)/2
    offset = nc.mean() / 3
    th = scipy.ndimage.filters.median_filter(nc, size=peak_filter_size) + offset
    peaks = []
    for i in xrange(1, nc.shape[0] - 5): #get rid of peak at end
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
    tempo,beats = librosa.beat.beat_track(y=music, sr=sample_rate, hop_length=hop_length)
    bsf = np.zeros((feature_vectors.shape[0],beats.size+1))
    # +1 because there can be music before first beat and after last beat
    for b in range(beats.size+1):
        if b ==0:
            temp = feature_vectors[:,0:beats[0]]
        elif b==beats.size:
            temp = feature_vectors[:,beats[b-1]:]
        else:
            temp = feature_vectors[:,beats[b-1]:beats[b]]

        # temp can be [] if beat tracker identifies beat at exact start or end of song
        # which results in nan when aggregator is called on it
        if temp.size == 0:
            temp = np.zeros((feature_vectors.shape[0],1))

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
    Returns: dictionary for each segment and its associated cluster as a number
    sim_mat: 2-D numpy array of similarity matrix
    bounds: indices of the boundaries indicating segment boundaries
    """
    size = len(bounds)
    bounds = [0] + bounds
    bounds = np.array(bounds)
    eigens = np.zeros((bounds.size,4))
    segments = np.zeros(bounds.size)
    for i in range(0,size):
        print bounds[i],(bounds[i+1]-bounds[i])/2
        eigen = np.linalg.svd(sim_mat[bounds[i]:int(bounds[i]+(bounds[i+1]-bounds[i])/2),bounds[i]:int(bounds[i]+(bounds[i+1]-bounds[i])/2)],full_matrices=0,compute_uv=False)

        #eigen = np.linalg.svd(sim_mat[bounds[i]:bounds[i]+16,bounds[i]:bounds[i]+16],full_matrices=0,compute_uv=False)
        eigens[i] = eigen[0:4]
        #going by the paper (below)
        #U,eigen,V = np.linalg.svd(sim_mat[bounds[i]:bounds[i+1],bounds[i]:bounds[i+1]],full_matrices=1,compute_uv=True)

        #test by first eigenvalues
    #dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(V.T,'euclidean'))
    #eigen_sim_mat = 1 - dist_mat/dist_mat.max()
    clusters = {}
    first_eigens = eigens[:,0]
    print first_eigens
    cluster_ctr = 0
    flag = 0
    #initialize
    clusters[0] = cluster_ctr
    for i in range(1,first_eigens.size):
        flag = 0
        print clusters
        for j in range(i):
            #compare everything before decimal, acceptance radius of 1
            if np.floor(np.abs(first_eigens[i] - first_eigens[j])) in [0,1]:
                clusters[i] = clusters[j]
                flag = 1
                break

        if not flag:
            cluster_ctr += 1
            clusters[i] = cluster_ctr
    print eigens
    print clusters
    return clusters

def KL_dist(mean_i, cov_i, mean_j,cov_j, num_features):
    A = 0.5*np.log(pdet(cov_j)/pdet(cov_i))
    B = 0.5*np.trace(np.multiply(cov_i,np.linalg.pinv(cov_j)))
    C = 0.5*((mean_i-mean_j).T).dot(np.linalg.pinv(cov_j).dot(mean_i-mean_j)) - num_features/2.0
    return A+B+C

def KL_dist_sym(mean_i,cov_i,mean_j,cov_j,num_features):
    A = np.trace(np.multiply(cov_i,np.linalg.pinv(cov_j))) + np.trace(np.multiply(cov_j,np.linalg.pinv(cov_i)))
    B_0 = np.dot((np.linalg.pinv(cov_i)+np.linalg.pinv(cov_j)),(mean_i-mean_j))
    B = np.dot((mean_i-mean_j).T,B_0)
    C = num_features
    D = 0.5*(A+B) - C
    return D

def D_seg(mean_i,cov_i,mean_j,cov_j,num_features):
    E = np.exp(-KL_dist_sym(mean_i,cov_i,mean_j,cov_j,num_features))
    return E

def pdet(mat):
    "pseudo_determinant"
    eig_values = np.linalg.svd(mat,full_matrices=1,compute_uv=False)
    pseudo_determinant = np.prod(eig_values[eig_values > 1e-8])
    return pseudo_determinant

def segment_cluster_better(stft,bounds,sr):
    segments = []
    num_features = stft.shape[0]
    #add 0 to bounds for looping ease
    bounds = [0] + bounds

    for i in range(len(bounds)):
        bounds[i] = int(np.floor(bounds[i]))
    #print len(bounds)
    #print bounds
    for i in range(len(bounds)):
        #print segments
        if i == len(bounds) - 1:
            segments.append(stft[:,bounds[i]:])
        else:
            #print "h",(stft[:,bounds[i]:bounds[i+1]]).shape
            segments.append(stft[:,bounds[i]:bounds[i+1]])

    mean = np.zeros((len(bounds),num_features))
    cov = np.zeros((len(bounds),num_features,num_features))
    #print cov.shape

    for i in range(len(bounds)):
        segments[i] = np.array(segments[i])
        mean[i] = np.mean(segments[i],axis=1)
        #cov[i] = np.cov(segments[i])

        for j in range(segments[i].shape[1]):
            temp_cov = np.zeros((segments[i].shape[1],num_features,num_features))
            temp_cov[i] = np.outer((segments[i][:,j]-mean[i]),(segments[i][:,j]-mean[i]))
        #print temp_cov[i]
        cov[i] = np.sum(temp_cov,axis = 0)
    segment_matrix = np.zeros((len(bounds),len(bounds)))
    print mean.shape, cov.shape, len(bounds), segment_matrix.shape
    #construct matrix
    for i in range(len(bounds)):
        for j in range(0,i+1): #to include i in the j range
            print "(%d,%d)" % (i,j)
            print "(%d,%d)" % (j,i)
            #print cov[i],cov[j]
            segment_matrix[i,j] = KL_dist_sym(mean[i],cov[i],mean[j],cov[j],num_features)
            segment_matrix[j,i] = KL_dist_sym(mean[i],cov[i],mean[j],cov[j],num_features)
    #segment_matrix = np.nan_to_num(segment_matrix)
    U,S,V = np.linalg.svd(segment_matrix,full_matrices=1,compute_uv=True)
    V = V.T

    m0 = S[0]*np.outer(U[:,0],V[:,0])
    m1 = S[1]*np.outer(U[:,1],V[:,1])
    m2 = S[2]*np.outer(U[:,2],V[:,2])
    m3 = S[3]*np.outer(U[:,3],V[:,3])
    m4 = S[4]*np.outer(U[:,4],V[:,4])
    m5 = S[5]*np.outer(U[:,5],V[:,5])
    m6 = S[6]*np.outer(U[:,6],V[:,6])

    plt.figure(0)
    plt.title('Matrix 0')
    plt.imshow(m0)
    plt.colorbar()

    plt.figure(1)
    plt.title('Matrix 1')
    plt.imshow(m1)
    plt.colorbar()

    plt.figure(2)
    plt.title('Matrix 2')
    plt.imshow(m2)
    plt.colorbar()


    plt.figure(3)
    plt.title('Matrix 3')
    plt.imshow(m3)
    plt.colorbar()

    plt.figure(4)
    plt.title('Matrix 4')
    plt.imshow(m4)

    plt.figure(5)
    plt.title('Matrix 5')
    plt.imshow(m5)

    plt.figure(6)
    plt.title('Matrix 6')
    plt.imshow(m6)
    plt.show()

def seconds_to_timestamp(seconds):
    minutes = math.floor(seconds / 60)
    return '%d:%02d' % (minutes, round(seconds - 60 * minutes))

def report_accuracy(identified_times, true_times_file):
    true_times = []
    with open(true_times_file, 'r') as f:
        for line in f:
            true_time = seconds_to_timestamp(float(line.split()[0]))
            true_times.append(true_time)

    print 'Identified times:'
    print identified_times
    print '\nTrue times:'
    print true_times

def filter_peaks(peaks, nc, peak_threshold_radius = 16):
    out_peaks = []
    i = 0
    while i < len(peaks):
        local_peak_vals = []
        local_peak_vals.append(nc[peaks[i]])
        j = i + 1
        while j < len(peaks) and peaks[j] - peaks[i] < peak_threshold_radius:
            local_peak_vals.append(nc[peaks[j]])
            j += 1
        argmax = local_peak_vals.index(max(local_peak_vals))
        out_peaks.append(peaks[i + argmax])
        i = j
    return out_peaks

def segment(music_path, feature_method = 'stft', distance_metric = 'euclidean', do_smooth = True, hop_length = 512, window_size = 2048, ker_size = 64, smoothing_window = 4, beat_sync = True, display = True, peak_threshold_radius = 16, peak_filter_size = 32):
    print "Loading..."
    music,sr = load_music(music_path)
    print "Featuring..."
    feature_vectorss,feature_vectors_beats = feature_vectors(music,sr,hop_length,window_size,feature_method=feature_method,beat_sync=beat_sync)
    print "Distancing..."
    sim_mat = sim_matrix(feature_vectors_beats,sr,hop_length,distance_metric=distance_metric,display=False)
    print "Noveling..."
    novelty_curve = compute_novelty_curve(sim_mat,ker_size=ker_size)

    if do_smooth:
        novelty_curve = smooth(novelty_curve,window_len = smoothing_window)
        novelty_curve = novelty_curve[3:] #hack

    peaks,th= pick_peaks(novelty_curve,peak_filter_size=peak_filter_size)
    peaks = filter_peaks(peaks, novelty_curve, peak_threshold_radius=peak_threshold_radius)
    #eigens = segment_cluster(sim_mat,peaks)
    tempo,beats = librosa.beat.beat_track(y=music,sr=sr,hop_length=hop_length)

    beat_times = np.zeros(beats.size)
    for i in range(beats.size):
        beat_times[i] = (beats[i]*hop_length)/float(sr)

    if display:

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(novelty_curve,color='g')
        ax1.plot(novelty_curve,color='r')
        ax1.plot(th)

        for p in peaks:
           ax1.axvline(x=p,color='m')

        regularTicks = 32 * np.arange(0, beat_times.size / 32)
        ax1.set_xticks(regularTicks)
        ax1.set_xticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])

        labelAx = ax1.twiny()
        labelAx.set_xlim(ax1.get_xlim())
        labelAx.set_xticks(peaks)
        labelAx.set_xticklabels([seconds_to_timestamp(beat_times[p]) for p in peaks])

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        plt.title('STFT')
        plt.imshow(sim_mat)
        ax2.set_xticks(regularTicks)
        ax2.set_xticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])
        ax2.set_yticks(regularTicks)
        ax2.set_yticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])
        plt.show()

    peaks = [beat_times[p] for p in peaks]
    #segment_cluster_better(feature_vectors_beats,peaks,sr)

    # add beginning and end of song to peaks
    if beat_times[0] not in peaks:
        peaks.insert(0, beat_times[0])
    if beat_times[-1] not in peaks:
        peaks.append(beat_times[-1])

    return peaks

#peaks = segment('LSD.m4a',display=True)

#segment('All_My_Friends.mp3')
# ##LOAD MUSIC, GET FEATURES, SIM MATRIX
# hop_length = 512
# window_size = 2048
# ker_size = 64
# smoothing_window = 4
#
# start = time.time()
# music,sr = load_music('All_My_Friends.mp3')
# true_times_file = 'Beatles/LSD.txt'
# end=time.time()
#
# print "Loading took %d seconds" % (end-start)
#
# start = time.time()
# feature_vectors = feature_vectors(music,sr,hop_length,window_size,method = 'stft',beat_sync = True)
# end = time.time()
#
# print "feature_vectors took %d seconds" % (end-start)
#
# start = time.time()
# sim_mat = sim_matrix(feature_vectors,sr,hop_length,distance_metric = 'euclidean',display=False)
# end=time.time()
#
# print "sim_mat took %d seconds" % (end-start)
#
# #GET NOVELTY CURVE OF SIMILARITY MATRIX
# start = time.time()
# novelty_curve = compute_novelty_curve(sim_mat,ker_size)
# end=time.time()
#
# print "novelty took %d seconds" % (end-start)
# #novelty_curve = novelty_curve[100:8000] #the end is wonky
#
# start=time.time()
# novelty_curve_smooth = smooth(novelty_curve,window_len = smoothing_window)
# novelty_curve_smooth = novelty_curve_smooth[3:]
# end=time.time()
#
# print "smoothing took %d seconds" % (end-start)
# #RECURRENCE MATRIX
# #recurr = librosa.segment.recurrence_matrix(feature_vectors,mode='affinity')
# #novelty_curve_recurr = compute_novelty_curve(recurr,ker_size/2)
# #novelty_curve_recurr = novelty_curve_recurr[100:8000] #the end is wonky
# #novelty_curve_recurr_smooth = smooth(novelty_curve_recurr,window_len = smoothing_window)
#
#
# #CALCULATE DERIVATIVE
# # deriv = np.zeros(novelty_curve_smooth.size)
# # for i in range(1,novelty_curve_smooth.size - 1):
# #     deriv[i-1] = (novelty_curve_smooth[i] - novelty_curve_smooth[i-1])
# # magnify = 10
# # deriv = magnify*deriv
#
# #PICK PEAKS
# start = time.time()
# # the smoothed novelty curve is a different length than the regular novelty curve?
# # causes index errors if a peak is picked at the end
# peaks,th= pick_peaks(novelty_curve_smooth)
# print peaks
# end = time.time()
#
# peaks = filter_peaks(peaks, novelty_curve_smooth, filter_radius=16)
#
# print "peak picking took %d seconds" % (end-start)
#
# start = time.time()
# print peaks
# eigens = segment_cluster(sim_mat,peaks)
# end = time.time()
# print "SVD took %d seconds" % (end-start)
# # peaks_recurr = pick_peaks(novelty_curve_recurr_smooth)
#
# #PRINT DERIVATIVE VALUES
# # deriv_pts = [deriv[i] for i in peaks]
# # print deriv_pts
#
#
# tempo,beats = librosa.beat.beat_track(y=music,sr=sr,hop_length=hop_length)
#
# beat_times = np.zeros(beats.size)
# for i in range(beats.size):
#     beat_times[i] = (beats[i]*hop_length)/float(sr)
#
# #PLOTTING
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111)
# ax1.plot(novelty_curve,color='g')
# ax1.plot(novelty_curve_smooth,color='r')
# ax1.plot(th)
# #plt.axhline(y=np.std(novelty_curve_smooth)/2,color='orange')
#
# #plt.plot(deriv,color='blue')
# for p in peaks:
#    ax1.axvline(x=p,color='m')
#
# regularTicks = 32 * np.arange(0, beat_times.size / 32)
# ax1.set_xticks(regularTicks)
# ax1.set_xticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])
#
# labelAx = ax1.twiny()
# labelAx.set_xlim(ax1.get_xlim())
# labelAx.set_xticks(peaks)
# labelAx.set_xticklabels([seconds_to_timestamp(beat_times[p]) for p in peaks])
#
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# plt.title('STFT')
# #plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# plt.imshow(sim_mat)
# ax2.set_xticks(regularTicks)
# ax2.set_xticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])
# ax2.set_yticks(regularTicks)
# ax2.set_yticklabels([seconds_to_timestamp(beat_times[i]) for i in regularTicks])
# end = time.time()
#
# # fig3 = plt.figure(3)
# # ax3 = fig3.add_subplot(111)
# # plt.title('ESM')
# # #plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# # plt.imshow(esm)
#
# print "plotting took %d seconds" % (end-start)
#
# report_accuracy([seconds_to_timestamp(beat_times[p]) for p in peaks], true_times_file)
#
# start = time.time()
# # plt.figure(3)
# # plt.title('RECURR')
# # plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# # plt.imshow(recurr)
# #
# # plt.figure(4)
# # plt.title('RECURR')
# # plt.plot(novelty_curve_recurr,color='g')
# # plt.plot(novelty_curve_recurr_smooth,color='r')
# # plt.xticks(np.arange(0, feature_vectors.shape[-1], skip), ['%.2f' % (i * hop_length / float(sr)) for i in range(feature_vectors.shape[-1])][::skip])
# # for p in peaks_recurr:
# #     plt.axvline(x=p,color='r')
# plt.show()
# end = time.time()
#
# print "showing took %d seconds" % (end-start)
