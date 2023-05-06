# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle


def dependency_adj_matrix(text, aspect, position, other_aspects):
    text_list = text.split()
    seq_len = len(text_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    position = int(position)
    flag = 1
    for i in range(seq_len):
        word = text_list[i]
        if (word in aspect and len(word)>1):
            for other_a in other_aspects:
                add = 0
                other_p = int(other_aspects[other_a])
                other_n = other_aspects.copy()
                del other_n[other_a]
                if other_n:
                    for other_l in other_n:
                        other_o = int(other_n[other_l])
                        other_e = other_n.copy()
                        del other_e[other_l]
                        if other_e:
                            for other_q in other_e:
                                other_g = int(other_aspects[other_q])
                        else:
                            other_g = 0
                            for other_w in other_a.split():
                                weight = 1 + (1 / (abs(position - (other_p + other_o * (other_o / (abs(other_o - other_p) \
                                           + abs(other_o - position) + abs(other_o - other_g))) + other_g * (other_g / \
                                            (abs(other_g - other_p) + abs(other_g - position) + abs(other_g - other_o))))) + 1))
                                matrix[i][other_p + add] = weight
                                matrix[other_p + add][i] = weight
                                add += 1
    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph_eh', 'wb')
    graph_idx = 0
    for i in range(len(lines)):
        aspects, polarities, positions, text = lines[i].split('\t')
        aspect_list = aspects.split('||')
        polarity_list = polarities.split('||')
        position_list = positions.split('||')
        text = text.lower().strip()
        aspect_graphs = {}
        aspect_positions = {}
        for aspect, position in zip(aspect_list, position_list):
            aspect_positions[aspect] = position
        for aspect, position in zip(aspect_list, position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            adj_matrix = dependency_adj_matrix(text, aspect, position, other_aspects)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph, fout)
    print('done !!!' + filename)
    fout.close()


if __name__ == '__main__':
    process('./datasets/rest14_train.raw')
    process('./datasets/rest14_test.raw')
    process('./datasets/lap14_train.raw')
    process('./datasets/lap14_test.raw')
    process('./datasets/rest15_train.raw')
    process('./datasets/rest15_test.raw')
    process('./datasets/rest16_train.raw')
    process('./datasets/rest16_test.raw')
    process('./datasets/twitter_train.raw')
    process('./datasets/twitter_test.raw')


