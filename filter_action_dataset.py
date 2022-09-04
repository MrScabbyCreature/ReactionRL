import time
# from utils import *
from rdkit import Chem
from action_utils import *

dataset = pd.read_csv("/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/action_dataset-updated.csv", index_col=0)

# Reset the index to be sequential
dataset.reset_index(inplace=True)
dataset = dataset.drop(columns=["index"])


# Init/Reset these columns
dataset["reactant_works"] = [True]*dataset.shape[0]
dataset["reactant_tested"] = [False]*dataset.shape[0]

dataset["action_works"] = [True] * dataset.shape[0]
dataset["action_tested"] = [False] * dataset.shape[0]


count = 0
t = time.time()
error_list = []

for i in range(0, 100):#dataset.shape[0]):
    in_mol = Chem.MolFromSmiles(dataset.iloc[i]["reactants"])
    dataset["reactant_tested"].iat[i] = True

    # Try out all the actions
    temp_df = dataset[dataset["rsig_clusters"].isin(get_applicable_rsig_clusters(in_mol))]
    if temp_df.shape[0] == 0:
        dataset["reactant_works"].iat[i] = False
    else:
        for j in range(temp_df.shape[0]):
            random_action = temp_df.iloc[j]
            dataset["action_tested"].at[random_action.name] = True
        
            # Try to apply action/
            try:
                apply_action(in_mol, random_action["rsub"], random_action["rcen"], random_action["rsig"], random_action["rsig_cs_indices"],
                                        random_action["psub"], random_action["pcen"], random_action["psig"], random_action["psig_cs_indices"])
            except Exception as e:
                error_list.append(type(e))
                dataset["action_works"].at[random_action.name] = False
            count += 1
            if count % 1000 == 0:
                print(count, time.time()-t, f"{dataset['reactant_tested'].sum()}({dataset.loc[dataset['reactant_tested']]['reactant_works'].sum()})", 
                                            f"{dataset['action_tested'].sum()}({dataset.loc[dataset['action_tested']]['action_works'].sum()})")
                t = time.time()
        # break

