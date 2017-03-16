from segment import segment
import fnmatch
import filecmp
import numpy as np
import os

def parse_ttl(dir_idx, song_idx):
    bounds = []
    types = []

    if dir_idx[0] != '0' and int(dir_idx) < 10:
        dir_idx = '0' + dir_idx
    if song_idx[0] != '0' and int(song_idx) < 10:
        song_idx = '0' + song_idx

    for dir in os.listdir('./The Beatles'):
        if dir[0:2] == dir_idx:
            dir_path = './The Beatles/' + dir
            for file in os.listdir(dir_path):
                if file[0:2] == song_idx:
                    label_path = './The Beatles/' + dir + '/' + file
                    audio_path = './The Beatles Songs/' + dir + '/' + file[0:len(file)-4] + '.mp3'
                    f = open(label_path)
                    for line in f:
                        bound_start_idx = 0
                        bound_end_idx = 0
                        bound_time = 0
                        bound_type = 0
                        for char_idx in range(len(line)):
                            #FOR TIME STAMP
                            if line[char_idx:char_idx+8] == 'beginsAt':
                                bound_start_idx = char_idx+12
                            if line[char_idx: char_idx+7] == 'S\"^^xsd':
                                bound_end_idx = char_idx
                            if(bound_start_idx and bound_end_idx):
                                bounds.append(line[bound_start_idx:bound_end_idx])
                                break

                            #FOR STRUCTURE TYPE
                            if line[char_idx:char_idx+10] == 'rdfs:label':
                                bound_type = line[char_idx+12:]
                                #print bound_type[0:bound_type.index('\"')]
                                types.append(bound_type[0:bound_type.index('\"')])
    for b in range(len(bounds)):

        if 'M' in bounds[b]:
            bounds[b] = 60*float(bounds[b][0:bounds[b].index('M')]) + float(bounds[b][bounds[b].index('M')+1:])
        else:
            bounds[b] = float(bounds[b])

    return np.floor(bounds), types, dir_path, label_path, audio_path

def bounds_check(pred_bounds,true_bounds,types,error_radius=5):
    total_num = len(true_bounds)
    print total_num
    correct_num = 0
    correct_types = []
    incorrect_types = []
    for t in range(len(true_bounds)):
        #exclude intro, outro, silence, and 0 bound
        if true_bounds[t] == 0:
            total_num -= 1
            continue

        incorrect_types.append(types[t])
        for b in pred_bounds:
            if np.abs(b-true_bounds[t]) <= error_radius:
                correct_num +=1
                incorrect_types = incorrect_types[0:len(incorrect_types)-1]
                correct_types.append(types[t])
                break

    return correct_num,total_num, correct_types, incorrect_types

def evaluate(dir_idx, song_idx, error_radius=5):
    true_bounds, types, dir_path, label_path, audio_path = parse_ttl(dir_idx, song_idx)
    pred_bounds = segment(audio_path, feature_method = 'stft', display=True)
    pred_bounds = np.floor(pred_bounds)
    print true_bounds, pred_bounds
    correct_num, total_num, correct_types, incorrect_types = bounds_check(pred_bounds, true_bounds, types, error_radius=error_radius)
    return correct_num, total_num, correct_types, incorrect_types

correct_num, total_num, correct_types, incorrect_types = evaluate('08','06')
print 'num correct:', correct_num, 'total num:', total_num,'correct types:', correct_types,'incorrect types:', incorrect_types
