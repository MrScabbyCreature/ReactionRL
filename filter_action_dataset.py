import time
# from utils import *
from rdkit import Chem
from action_utils import *
from multiprocessing import Process, Manager

dataset = pd.read_csv("/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/action_dataset-updated.csv", index_col=0)

# Reset the index to be sequential
dataset.reset_index(inplace=True)
dataset = dataset.drop(columns=["index"])


# Init/Reset these columns
dataset["reactant_works"] = [True]*dataset.shape[0]
dataset["reactant_tested"] = [False]*dataset.shape[0]

dataset["action_works"] = [True] * dataset.shape[0]
dataset["action_tested"] = [False] * dataset.shape[0]

def find_bad_action_index(in_mol, temp_df):
    bad_actions = []
    for j in range(temp_df.shape[0]):
        action = temp_df.iloc[j]
    
        # Try to apply action/
        try:
            apply_action(in_mol, action["rsub"], action["rcen"], action["rsig"], action["rsig_cs_indices"],
                                    action["psub"], action["pcen"], action["psig"], action["psig_cs_indices"])
        except Exception as e:
            bad_actions.append(action.name)
    return bad_actions



if __name__ == "__main__":
    count = 0
    t = time.time()
    # manager = Manager()
    # result_q = manager.Queue()

    for i in range(0, 10):#dataset.shape[0]):
        in_mol = Chem.MolFromSmiles(dataset.iloc[i]["reactants"])
        dataset["reactant_tested"].iat[i] = True

        # Try out all the actions
        temp_df = dataset[dataset["rsig_clusters"].isin(get_applicable_rsig_clusters(in_mol))]
        if temp_df.shape[0] == 0:
            dataset["reactant_works"].iat[i] = False
        else:
            # Tested's
            dataset.loc[temp_df.index, "action_tested"] = True
            # Worked's
            for action in find_bad_action_index(in_mol, temp_df):
                dataset["action_works"].at[action] = False
        print(i, time.time()-t, f"{dataset['reactant_tested'].sum()}({dataset.loc[dataset['reactant_tested']]['reactant_works'].sum()})", 
                                    f"{dataset['action_tested'].sum()}({dataset.loc[dataset['action_tested']]['action_works'].sum()})")
        t = time.time()

    print("Dumping...")
    dataset.to_csv("/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/action_dataset-filtered.csv")