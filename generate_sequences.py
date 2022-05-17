import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Pool
import pickle

# load data
dataset = pd.read_csv("/home/abhor/Desktop/datasets/my_uspto/processed_data.csv", index_col=0)

X = dataset["reactants"]
Y = dataset["products"]

adjacency_dict = {smile: [] for smile in set(X).union(set(Y))}

for row in zip(X, Y):
    adjacency_dict[row[0]].append(row[1])

batch_size = 100
unique_X = np.unique(list(set(X) - set(Y))) # only do it for nodes that do not have other nodes leading to it (this gives starting points for sequences)
max_len = len(unique_X)

def func(start_idx):

    def get_sequences(starting_SMILE, seen_strings=[]):
        '''
        starting_SMILE: index in Y to start from
        '''
        seen_strings = list(seen_strings) # without this, somehow the function remembers seen_strings across calls
        if len(adjacency_dict[starting_SMILE]) == 0:
            return [[starting_SMILE]]

        if seen_strings == []: # if empty, means we just started. So add the starting SMILE string
            seen_strings.append(starting_SMILE)
        
        sequences = []
        for next_SMILE in adjacency_dict[starting_SMILE]:
            if next_SMILE in seen_strings: # if we have already seen this string, means loop -> stop
                return [[starting_SMILE]]
            else:
                seen_strings.append(next_SMILE)
            
            next_sequences = get_sequences(next_SMILE, list(seen_strings))
            for s in next_sequences:
                sequences.append([starting_SMILE]+s)
        return sequences

    listy = []
    max_len = len(unique_X)
    for i in range(start_idx, min(start_idx+batch_size, max_len)):
        listy.extend(get_sequences(unique_X[i]))
    return listy

if __name__ == "__main__":
    # doing in multiprocessing batches, otherwise too slow
    pool = Pool(16) # may want to change to number of cores in the system
    reaction_sequence = list(tqdm.tqdm(pool.imap_unordered(func, range(0, max_len, batch_size)), total=max_len//batch_size))
    pickle.dump(reaction_sequence, open("/home/abhor/Desktop/datasets/my_uspto/intermediate_sequences.pickle", 'wb'))