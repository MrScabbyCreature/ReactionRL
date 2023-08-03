from offlineRL_utils import *
import argparse, glob
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

def get_topk_predictions(model, source_list, target_list, topk=10):
    # Convert to mols
    if isinstance(source_list, pd.Series):
        source_list = source_list.tolist()
    if isinstance(target_list, pd.Series):
        target_list = target_list.tolist()
    tt = time.time()
    sources = data.Molecule.pack(list(map(molecule_from_smile, source_list)))
    targets = data.Molecule.pack(list(map(molecule_from_smile, target_list)))
    print(f"Took {time.time() - tt}s to pack molecules.")

    # Predictions
    batch_size = 1024
    pred = torch.concatenate([model(sources[i:min(i+batch_size, sources.batch_size)].to(device), 
                                 targets[i:min(i+batch_size, sources.batch_size)].to(device), None, None, "actor").detach() for i in range(0, sources.batch_size, batch_size)], axis=0)

    action_embeddings = get_action_dataset_embeddings(model.GIN, action_rsigs, action_psigs)

    # Get applicable actions for source(s)
    applicable_action_indices_list = []
    
    with Pool(30) as p:
        for idxes in tqdm.tqdm(p.imap(functools.partial(get_emb_indices_and_correct_idx, no_correct_idx=True), 
                                      [{"reactant": source_list[i]} for i in range(pred.shape[0])], chunksize=10),
                              total=pred.shape[0]):
            applicable_action_indices_list.append(idxes)

    # Sort by critic's Q
    dict_of_list_of_indices = {}
    
    for i in tqdm.tqdm(range(pred.shape[0])):
        pred_for_i = pred[i]
        adi = applicable_action_indices_list[i]
        if len(adi) == 0:
            dict_of_list_of_indices[i] = np.array([])
            continue

        # Get top 50 for actor
        # dist = torch.linalg.norm(action_embeddings[adi] - pred[i], axis=1) # USE COS DISTANCE
        dist = 1 - torch.mm(action_embeddings[adi], pred[i].view(-1, 1)).view(-1)/(torch.linalg.norm(action_embeddings[adi], axis=1)*torch.linalg.norm(pred[i]))

        dict_of_list_of_indices[i] = adi[torch.argsort(dist)[:50].cpu().numpy().astype(np.int64)]

    # Sort with critic's Q
    i_sorted = list(range(pred.shape[0]))
    action_indices = np.concatenate([dict_of_list_of_indices[i] for i in i_sorted])
    state_indices = np.concatenate([np.full_like(dict_of_list_of_indices[i], i) for i in i_sorted])
    critic_qs = []
    for i in tqdm.tqdm(range(0, action_indices.shape[0], batch_size)):
        batch_reactants = sources[state_indices[i:i+batch_size]]
        batch_products = targets[state_indices[i:i+batch_size]]
        batch_rsigs = action_rsigs[action_indices[i:i+batch_size]]
        batch_psigs = action_psigs[action_indices[i:i+batch_size]]
        critic_qs.append(ac(batch_reactants.to(device), batch_products.to(device), batch_rsigs.to(device), batch_psigs.to(device), "critic").detach().cpu().numpy())

    critic_qs = np.concatenate(critic_qs)

    # Get action predictions
    action_pred_indices = []
    start = 0
    for i in tqdm.tqdm(i_sorted):
        end = start + dict_of_list_of_indices[i].shape[0]
        i_critic_qs = critic_qs[start:end]

        action_pred_indices.append(dict_of_list_of_indices[i][i_critic_qs.reshape(-1).argsort()[::-1]][:topk])
        start = end

    return action_pred_indices

def apply_actions_on_reactant(args):
    reactant, action_dataset_idx = args
    listy = []
    for idx in action_dataset_idx:
        try:
            listy.append(Chem.MolToSmiles(apply_action(Chem.MolFromSmiles(reactant), *action_dataset.iloc[idx])))
        except Exception as e:
            pass
    return listy

def get_similarity_from_smiles(s1, s2):
    return similarity(Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2))

def get_args():
    parser = argparse.ArgumentParser()
    # Model and cuda
    parser.add_argument("--model-path", type=str, required=True, help="Which model to run for")
    parser.add_argument("--cuda", type=int, default=-1, help="Run on GPU? If yes, which?")
    
    # Args for loading data
    parser.add_argument("--dump-test-data", action="store_true", help="Dump test data (overwrite if doesn't exist)")
    parser.add_argument("--load-step", type=int, required=True, help="Which step to run for")
    parser.add_argument("--N", type=int, required=True, help="How many test samples to run evaluation for")
    
    # Args for evaluation
    parser.add_argument("--branching-factor", type=int, required=True, help="Branching factor for evaluation (--bf per mol)")
    parser.add_argument("--eval-for-steps", type=int, required=True, help="How many steps to evaluate for")
    parser.add_argument("--top-similar-k-per-step-branch", type=int, default=None, help="Per branch step, select the top few steps to furhter evaluate")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # if args.load_step in [5, 10] and args.top_similar_k_per_step_branch is None:
    #     exit()
    print()
    print()
    print("#######################################################")
    print("######################## Start ########################")
    print("#######################################################")
    print(args)

    start_time = time.time()
    calc_start_mol_prob_dist()

    ##############
    # Set device #
    ##############
    device = set_device(torch.device("cpu" if not (torch.cuda.is_available() or args.cuda == -1) else f"cuda:{args.cuda}"))
    print("Running on device:", device)

    ###############################
    # dump/load data as requested #
    ###############################
    # test_file = "models/supervised/trajectory_test_data.csv"
    # if args.dump_test_data:
    #     test_data_df = generate_train_data(N=200_000, steps=10) # Generate 20,000 trajectories of max length 10 and dump
    #     test_data_df.to_csv(test_file)
    # else:
    #     test_data_df = pd.read_csv(test_file, index_col=0)
    # test_data_df = test_data_df[test_data_df['step']==args.load_step]
    # assert test_data_df.shape[0] >= args.N, f"Requested {args.N} samples. Found max of {test_data_df.shape[0]}. Change dump stats in code...."
    main_df_dict = pickle.load(open("models/supervised/evaluation_dict.pickle", 'rb'))
    test_data_df = main_df_dict[args.load_step].iloc[:args.N]
    source_list, target_list = test_data_df["reactant"].values, test_data_df["product"].values
    print(f"Source list shape = {source_list.shape} || target list shape = {target_list.shape}")

    #####################################
    # Getting action dataset signatures #
    #####################################
    tt = time.time()
    action_rsigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["rsig"])))
    action_psigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["psig"])))
    print(f"Took {time.time() - tt:.2f}s to load action signatures into memory.")

    ##############
    # Load model #
    ##############
    ac = torch.load(args.model_path).to(device)
    
    ##############################
    # Get trajectory predictions #
    ##############################
    target_list_idx = np.arange(target_list.shape[0])
    sim_dict = {}
    trajectory_dict = {str(i): source_list[i] for i in range(len(source_list))} # Keeps track of trajectory in dict format (need hash keeys for quick access)
    source_keys = list(map(str, np.arange(len(source_list)))) # Map for index to keys of previous step in trajectory

    # RUN -----------------------
    for i_step in range(1, args.eval_for_steps+1): 
        print("Running prediction for step", i_step)
        # Get action predictions
        pred = get_topk_predictions(ac, source_list, target_list[target_list_idx], topk=args.branching_factor)

        # get products
        temp_source_keys = []
        temp_source_list = []
        with Pool(30) as p:
            print("Applying actions for step", i_step)
            for i, product_list in tqdm.tqdm(enumerate(p.imap(apply_actions_on_reactant, zip(source_list, pred), chunksize=10)), total=len(pred)):
                for _i, product in enumerate(product_list):
                    key = f"{source_keys[i]}_{_i}"
                    trajectory_dict[key] = product
                    sim_dict[key] = get_similarity_from_smiles(product, target_list[int(key.split("_")[0])])
                    temp_source_keys.append(key)
                    temp_source_list.append(product)
                    
        print("Getting top some sim products for each s-t pair")
        temp_source_keys = np.array(temp_source_keys)
        temp_source_list = np.array(temp_source_list)
        temp_source_sim = np.vectorize(sim_dict.get)(temp_source_keys)
        temp_source_argsort = np.argsort(temp_source_sim)
        temp_source_st_idx = np.array(list(map(lambda x: int(x.split("_")[0]), temp_source_keys)))
        temp_source_indices = []
        for t_i in range(target_list.shape[0]):
            if args.top_similar_k_per_step_branch is not None:
                temp_source_indices.append((temp_source_argsort[temp_source_st_idx == t_i])[:args.top_similar_k_per_step_branch])
            else:
                temp_source_indices.append((temp_source_argsort[temp_source_st_idx == t_i]))

        temp_source_indices = np.concatenate(temp_source_indices)
        
        # update source list and source_keys for next step
        source_keys = temp_source_keys[temp_source_indices]
        source_list = temp_source_list[temp_source_indices]
        target_list_idx = list(map(lambda x: int(x.split("_")[0]), source_keys))

    ############
    # Analysis #
    ############
    # Save separate keys for steps and s-t pairs (for conv processing later)
    all_keys = np.array(list(trajectory_dict.keys()))
    stepwise_keys = {}
    for step in tqdm.tqdm(range(0, args.eval_for_steps+1)):
        stepwise_keys[step] = set(list(filter(lambda x: len(x.split('_')) == step+1, all_keys)))

    source_index_wise_keys = {}
    for i in tqdm.tqdm(range(len(target_list))):
        source_index_wise_keys[i] = set(list(filter(lambda x: x == str(i) or x.startswith(str(i)+'_'), all_keys)))

    best_sim_list = []
    step_for_best_sim_list = []

    for i in tqdm.tqdm(range(len(target_list))):
        keys_for_i = list(filter(lambda x: len(x.split("_")) > 1, list(source_index_wise_keys[i])))
        sim_list_for_i = np.vectorize(sim_dict.get)(keys_for_i)
        
        best_arg = sim_list_for_i.argmax()
        best_sim_list.append(sim_list_for_i[best_arg])
        step_for_best_sim_list.append(len(keys_for_i[best_arg].split("_"))-1)

    best_sim_list = np.array(best_sim_list)
    step_for_best_sim_list = np.array(step_for_best_sim_list)
    print(f"Average best sim = {np.mean(best_sim_list)}")
    print(f"% correct paths predicted = {(best_sim_list==1).sum()/best_sim_list.shape[0]*100}")
    print(f"Average path length for best sim = {np.mean(step_for_best_sim_list)}")


    directory = 'models/supervised/traj_eval/'
    directory += str(len(glob.glob(os.path.join(directory, '*'))))
    os.makedirs(directory)

    ################
    # Save Results #
    ################
    # Save args + metrics
    out_dict = vars(args)
    out_dict.update(
        {
            "average best sim": np.mean(best_sim_list),
            "% correct paths predicted": (best_sim_list==1).sum()/best_sim_list.shape[0]*100,
            "average path length for best sim": np.mean(step_for_best_sim_list),
            "total_time": round(time.time() - tt, 2),
        }
    )
    with open(os.path.join(directory, "args_and_results.json"), 'w') as f:
        json.dump(out_dict, f, indent=4)

    # Save traj_dict - Everything else can be calculated again from that if needed for checking
    pickle.dump(trajectory_dict, open(os.path.join(directory, "trajectory_dict.pickle"), 'wb'))

    # Plot 1 - All similarities
    sim_vals = np.array(list(sim_dict.values()))
    sns.displot(sim_vals, kde=True, bins=50)
    plt.title("All similarities")
    plt.xlabel("tanimoto sim over morgan fp")
    plt.savefig(os.path.join(directory, "all_sim.png"))

    # Plot 2 - Best similarities
    sns.displot(best_sim_list, kde=True, bins=50)
    plt.title("Best similarities")
    plt.xlabel("tanimoto sim over morgan fp")
    plt.savefig(os.path.join(directory, "best_sim.png"))

    # Plot 3 - step-wise best sim 
    plt.figure(figsize=(8, 8))
    sns.histplot(pd.DataFrame({"steps": step_for_best_sim_list, "sim": best_sim_list}), x="sim", hue='steps', multiple='dodge', shrink=0.7, palette=sns.color_palette())
    plt.ylabel("step count")
    plt.xlabel("Similarity")
    plt.title("Step count at different similarity values for best paths")
    plt.savefig(os.path.join(directory, "stepwise_best_sim.png"))