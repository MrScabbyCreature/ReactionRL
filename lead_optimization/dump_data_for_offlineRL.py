import argparse, tqdm
from action_utils import apply_action, get_applicable_actions
from rdkit import Chem
import pandas as pd, numpy as np
from multiprocessing import Pool, cpu_count
import functools

def _get_app_act_count(smile):
    act = get_applicable_actions(Chem.MolFromSmiles(smile))
    if len(act.shape) > 0:
        return act.shape[0]
    return 0


def calc_start_mol_prob_dist(processes=5):
    # Probabilities for start mol sampling
    print("Calculating probability for start mol sampling")
    applicable_action_count = []
    with Pool(processes) as p:
        for c in tqdm.tqdm(p.imap(_get_app_act_count, start_mols, chunksize=100), total=len(start_mols)):
            applicable_action_count.append(c)
    
    applicable_action_count = np.array(applicable_action_count)
    return  applicable_action_count / applicable_action_count.sum()

def _generate_data(smile, steps):
    mol = Chem.MolFromSmiles(smile)

    df = pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product', 'step'])
    index = []
    
    # Get sequences
    try:
        for i in range(steps):
            actions = get_applicable_actions(mol)
            if actions.shape[0] == 0:
                raise Exception("No actions applicable.....")

            # Apply a random action
            rand_idx = np.random.randint(0, actions.shape[0])
            product = apply_action(mol, *actions.iloc[rand_idx])

            # Add it to df
            df.loc[df.shape[0], :] = [Chem.MolToSmiles(mol)] + actions.iloc[rand_idx].tolist() + [Chem.MolToSmiles(product), i]
            index.append(actions.iloc[rand_idx].name)

            # Next reactant = product
            mol = product
    except Exception as e:
        return pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product', 'step'])
    
    # Fix index
    df.index = index
    
    # Fix target
    df["product"] = Chem.MolToSmiles(product)

    # Fix steps
    df["step"] = df.shape[0] - df["step"]

    return df

def generate_data(N, steps, processes=5):
    # Generate dataset
    df_list = []
    final_shape = 0
    smiles_per_random_sample = 1000

    # Create dataset for multi-step pred
    print("Creating dataset...")
    if processes > 1:
        with Pool(processes) as p, tqdm.tqdm(total=N) as pbar:
            while final_shape < N:
                smiles = np.random.choice(start_mols, size=(smiles_per_random_sample,), p=start_mol_prob)

                for new_df in p.imap_unordered(functools.partial(_generate_data, steps=steps), smiles, chunksize=10):
                    df_list.append(new_df)
                    final_shape += new_df.shape[0]

                pbar.update(final_shape - pbar.n)
    else:
        raise Exception("Single process not implemented... :'(")

    main_df = pd.concat(df_list)
    return main_df

def get_args():
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100000, help="Number of data points to use")
    parser.add_argument("--steps", type=int, required=True, help="Number of data points to use")
    parser.add_argument("--processes", type=int, default=int(0.8*cpu_count()), help="Number of CPU cores to use for multiprocessing")
    return parser.parse_args()

if __name__ == "__main__":
    import pickle
    start_mols = pickle.load(open("datasets/my_uspto/unique_start_mols.pickle", 'rb'))
    args = get_args()

    start_mol_prob = calc_start_mol_prob_dist()

    # Dump train and test (test is 20% of train samples)
    for data_type in ["train", "test"]:
        file = f"datasets/offlineRL/{args.steps}steps_{data_type}.csv"
        samples = args.train_samples if data_type == "train" else int(0.2 * args.train_samples)
        df = generate_data(samples, args.steps, processes=args.processes)
        df.to_csv(file)
        print(f"Dumped at {file}. Shape = {df.shape}")