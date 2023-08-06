from offlineRL_utils import *
import argparse
from action_utils import dataset as action_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Hyperparams
actor_lr = 3e-4
critic_lr = 1e-3
epochs = 50
batch_size = 128
topk = 10


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True, help="Which step data to load")
parser.add_argument("--model-type", type=str, choices=["actor", "critic", "actor-critic"], required=True, help="Type of model to train")
parser.add_argument("--actor-loss", type=str, choices=["mse", "PG"], default=None, help="Actor loss")
parser.add_argument("--negative-selection", type=str, choices=["random", "closest"], default=None, help="Actor loss")
parser.add_argument("--cuda", type=int, required=True, help="Which device to use")
args = parser.parse_args()

# Device
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load data
train_df = pd.read_csv(f"datasets/offlineRL/{args.steps}steps_train.csv", index_col=0).sample(frac=1)

# Init model
if args.model_type == "actor":
    train_actor = True
    train_critic = False
    model = ActorNetwork()
    
if args.model_type == "critic":
    train_actor = False
    train_critic = True
    model = CriticNetwork()
    
if args.model_type == "actor-critic":
    train_actor = True
    train_critic = True
    model = ActorCritic()
model = model.to(device)

# Pack action signatures for fast embeddings
action_rsigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["rsig"])))
action_psigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["psig"])))

# Store correct indices for fast metric calc
correct_applicable_indices = []
correct_action_dataset_indices = []
action_embedding_indices = []

with Pool(35) as p:
    for indices_used_for_data, correct_app_idx, correct_act_idx in p.imap(get_emb_indices_and_correct_idx, train_df.iterrows(), chunksize=50):
        action_embedding_indices.append(indices_used_for_data)
        correct_applicable_indices.append(correct_app_idx)
        correct_action_dataset_indices.append(correct_act_idx)

# Train and valid data
train_idx = np.arange(0, int(train_df.shape[0]*0.8))
valid_idx = np.arange(int(train_df.shape[0]*0.8), train_df.shape[0])

train_reactants = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[train_idx]["reactant"]))).to(device)
train_products = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[train_idx]["product"]))).to(device)
train_rsigs = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[train_idx]["rsig"]))).to(device)
train_psigs = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[train_idx]["psig"]))).to(device)

valid_reactants = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[valid_idx]["reactant"]))).to(device)
valid_products = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[valid_idx]["product"]))).to(device)
valid_rsigs = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[valid_idx]["rsig"]))).to(device)
valid_psigs = data.Molecule.pack(list(map(molecule_from_smile, train_df.iloc[valid_idx]["psig"]))).to(device)

print("Train and valid data shapes:")
print(train_reactants.batch_size, train_products.batch_size, valid_reactants.batch_size, valid_products.batch_size)
if train_critic:
    print(train_rsigs.batch_size, train_psigs.batch_size, valid_rsigs.batch_size, valid_psigs.batch_size)

# Optimizers
if train_actor:
    actor_optimizer = torch.optim.Adam(model.parameters(), lr=actor_lr)  
if train_critic:
    critic_optimizer = torch.optim.Adam(model.parameters(), lr=critic_lr)  
critic_loss_criterion = nn.MSELoss()

# For actor log prob calc
if train_actor:
    actor_log_std = nn.Parameter(torch.zeros(model.actor.last_layer.out_features, dtype=torch.float32)).to(device)

# Embeddings init
embedding_model = torch.load("models/zinc2m_gin.pth").to(device)
embedding_model.load_state_dict(model.GIN.state_dict())
action_embeddings = get_action_dataset_embeddings(embedding_model, action_rsigs, action_psigs)
action_embeddings_norm = torch.linalg.norm(action_embeddings, axis=1)

# Some helper inits
best_rank = 10000
best_metric = -100
best_model = None
actor_metric_dict = {"cos_rank_mean": [], "euc_rank_mean": [], "cos_rank_std": [], "euc_rank_std": [], 
               "cos_rank_tot": [], "euc_rank_tot": [], "rmse": [], "cos_sim": [], "time(epoch_start-now)": []}
critic_metric_dict = {"GT_acc": [], "GT_rec": [], "GT_prec": [], "GT_f1": [], 
                "others_acc": [], "others_rec": [], "others_prec": [], "others_f1": [], 
                "mean_acc": [], "mean_rec": [], "mean_prec": [], "mean_f1": [],  "time(epoch_start-now)": []}

###############################################################################################
####################################### Train the model #######################################
###############################################################################################
print(args)
for epoch in range(1, epochs+1):
    start_time = time.time()
    model.train()
    for i in range(0, train_reactants.batch_size - batch_size, batch_size):
        # Forward pass
        actor_actions = model(train_reactants[i:i+batch_size], train_products[i:i+batch_size], train_rsigs[i:i+batch_size], train_psigs[i:i+batch_size], "actor")

        if train_critic or args.actor_loss == "PG":
            # Calc negatives
            negative_indices = []

            for _i in range(actor_actions.shape[0]):
                correct_action_dataset_index = correct_action_dataset_indices[train_idx[i+_i]]
                if args.negative_selection == "random":
                    size = min(topk, action_embedding_indices[train_idx[i+_i]].shape[0])
                    negative_indices.append(np.random.choice(action_embedding_indices[train_idx[i+_i]], size=(size,), replace=False))
                if args.negative_selection == "closest":
                    curr_out = actor_actions[_i].detach()
                    dist = torch.linalg.norm(action_embeddings - curr_out, axis=1)
                    sorted_idx = torch.argsort(dist)[:topk] # get topk
                    sorted_idx = sorted_idx[sorted_idx != correct_action_dataset_index] # Remove if correct index in list
                    negative_indices.append(sorted_idx)

        # critic update
        if train_critic:
            batch_reactants = train_reactants[sum([[i+_i]*(1+negative_indices[_i].shape[0]) for _i in range(actor_actions.shape[0])], [])]
            batch_products = train_products[sum([[i+_i]*(1+negative_indices[_i].shape[0]) for _i in range(actor_actions.shape[0])], [])]
            batch_rsigs = action_rsigs[sum([[correct_action_dataset_indices[train_idx[i+_i]]] + negative_indices[_i].tolist() for _i in range(actor_actions.shape[0])], [])]
            batch_psigs = action_psigs[sum([[correct_action_dataset_indices[train_idx[i+_i]]] + negative_indices[_i].tolist() for _i in range(actor_actions.shape[0])], [])]
            batch_q_targets = torch.Tensor(sum([[1] + [0] * negative_indices[_i].shape[0] for _i in range(actor_actions.shape[0])], [])).view(-1, 1)


            critic_qs = model(batch_reactants.to(device), batch_products.to(device), batch_rsigs.to(device), batch_psigs.to(device), "critic")
            critic_loss = critic_loss_criterion(critic_qs, batch_q_targets.to(device))
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        # actor update
        if train_actor:
            actor_actions = model(train_reactants[i:i+batch_size], train_products[i:i+batch_size], train_rsigs[i:i+batch_size], train_psigs[i:i+batch_size], "actor")
            if args.actor_loss =="mse":
                actor_loss = nn.MSELoss()(actor_actions, 
                                        get_action_embedding_from_packed_molecule(embedding_model,
                                                                                    train_rsigs[i:i+batch_size], 
                                                                                    train_psigs[i:i+batch_size]))
            elif args.actor_loss == "PG":
                normal_dist = torch.distributions.Normal(actor_actions, actor_log_std.exp())
                positives = get_action_embedding_from_packed_molecule(embedding_model, train_rsigs[i:i+batch_size], train_psigs[i:i+batch_size])
                positive_log_pi = normal_dist.log_prob(positives)
                negative_log_pi = []
                for _i, _indices in enumerate(negative_indices):
                    normal_dist = torch.distributions.Normal(actor_actions[_i], actor_log_std.exp())
                    negative_log_pi.append(normal_dist.log_prob(action_embeddings[_indices]))
                negative_log_pi = torch.concatenate(negative_log_pi, axis=0)

                actor_loss = torch.concatenate([-positive_log_pi, (1/(topk*2))*negative_log_pi], axis=0).sum(-1, keepdim=True).mean() # Using R = 1 for positives, and R = -1/2topk for negatives 
            else:
                raise Exception(f"Unexpected actor loss {args.actor_loss}")

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        # Emptry any cache (free GPU memory)
        torch.cuda.empty_cache()
    
    result_string = f'Epoch {epoch}/{epochs}. Batch {i}/{train_reactants.batch_size - batch_size}.'
    if train_actor:
        result_string +=  f'Actor loss = {actor_loss.item():.6f}'
    if train_critic:
        result_string += f' || critic loss = {critic_loss.item():.6f}'
    print (result_string)
    

    # VALIDATION
    model.eval()
    with torch.no_grad():
        d = vars(args)
        margin_string = "# " + " || ".join([f"{k}--{d[k]}" for k in d]) + " #"
        print("#" * len(margin_string))
        print(margin_string)
        print("#" * len(margin_string))

        if train_actor: # Display actor metrics
            # Predictions and action component-wise loss
            pred = torch.concatenate([model(valid_reactants[i:i+batch_size], 
                                            valid_products[i:i+batch_size], 
                                            valid_rsigs[i:i+batch_size], 
                                            valid_psigs[i:i+batch_size], "actor").detach() \
                                    for i in range(0, valid_reactants.batch_size-batch_size, batch_size)], axis=0)
            true = get_action_embedding_from_packed_molecule(embedding_model, valid_rsigs[:pred.shape[0]], valid_psigs[:pred.shape[0]]) 

            metric_df = pd.DataFrame(columns=["rmse", "cos_sim", "euc_rank_mean", "euc_rank_std", "euc_rank_tot", "cos_rank_mean", "cos_rank_std", "cos_rank_tot", "time(epoch_start-now)"])

            # Print valid metrics
            actor_metric_dict["rmse"].append( (((pred-true)**2).sum(axis=1)**0.5).mean().item() )
            actor_metric_dict["cos_sim"].append( ((pred*true).sum(axis=1) / torch.linalg.norm(pred, axis=1) / torch.linalg.norm(true, axis=1)).mean().item() )

            # Print valid metric - Rank
            for dist in ["euclidean", "cosine"]:
                rank_list = []
                l = []
                total = []
                for i in range(pred.shape[0]):
                    pred_for_i = pred[i]
                    act_emb_for_i, correct_applicable_index = action_embeddings[action_embedding_indices[valid_idx[i]]], correct_applicable_indices[valid_idx[i]]

                    rank, list_of_indices = get_ranking(pred_for_i, act_emb_for_i, correct_applicable_index, distance=dist)
                    l.append(rank.item())
                    total.append(act_emb_for_i.shape[0])
                rank_list.append(f"{np.mean(l):.4f}({np.mean(total)}) +- {np.std(l):.4f}")
                actor_metric_dict[f"{dist[:3]}_rank_mean"].append(np.mean(l))
                actor_metric_dict[f"{dist[:3]}_rank_std"].append(np.std(l))
                actor_metric_dict[f"{dist[:3]}_rank_tot"].append(np.mean(total))

            actor_metric_dict["time(epoch_start-now)"].append(f"{(time.time()-start_time)/60:.2f} min")
            for col in metric_df.columns:
                metric_df[col] = [actor_metric_dict[col][-1]]
            metric_df.index = [epoch]
            print(tabulate(metric_df, headers='keys', tablefmt='fancy_grid'))
            print()

            # Update best model
            if actor_metric_dict["euc_rank_mean"][-1] < best_rank:
                best_rank = actor_metric_dict["euc_rank_mean"][-1]
                best_model = type(model)()
                best_model.load_state_dict(model.state_dict())
                best_epoch = epoch
                print(f"BEST MODEL UPDATED! BEST RANK = {best_rank}")
        
        if train_critic and not train_actor: # Display critic metrics - only if no actor
            # Predict for GT
            GT_pred_qs = (torch.concatenate([model(valid_reactants[i:i+batch_size], 
                        valid_products[i:i+batch_size], 
                        valid_rsigs[i:i+batch_size], 
                        valid_psigs[i:i+batch_size], 
                        "critic").detach() for i in range(0, valid_reactants.batch_size-batch_size, batch_size)], axis=0).cpu().numpy() > 0.5).astype(int)
            GT_true_qs = np.ones_like(GT_pred_qs)

            # Pred for others
            negative_indices = []

            for i in valid_idx:
                correct_action_dataset_index = correct_action_dataset_indices[i]
                curr_out = action_embeddings[correct_action_dataset_index]
                dist = torch.linalg.norm(action_embeddings - curr_out, axis=1)

                # Get the closest that is not GT
                sorted_idx = torch.argsort(dist)[:2]
                sorted_idx = sorted_idx[sorted_idx != correct_action_dataset_index] # Remove if correct index in list
                sorted_idx = sorted_idx[:1]
                negative_indices.append(sorted_idx)

            valid_batch_reactants = valid_reactants[sum([[i]*negative_indices[i].shape[0] for i in range(valid_idx.shape[0])], [])].to(device)
            valid_batch_products = valid_products[sum([[i]*negative_indices[i].shape[0] for i in range(valid_idx.shape[0])], [])].to(device)
            valid_batch_rsigs = action_rsigs[torch.concatenate(negative_indices)].to(device)
            valid_batch_psigs = action_psigs[torch.concatenate(negative_indices)].to(device)

            others_pred_qs = (torch.concatenate([model(valid_batch_reactants[i:i+batch_size], 
                        valid_batch_products[i:i+batch_size], 
                        valid_batch_rsigs[i:i+batch_size], 
                        valid_batch_psigs[i:i+batch_size],
                        "critic").detach() for i in range(0, valid_batch_reactants.batch_size-batch_size, batch_size)], axis=0).cpu().numpy() > 0.5).astype(int)
            others_true_qs = np.zeros_like(others_pred_qs)

            # Update metrics (with inverted labels -- sklearn considers 0 as true class in confusion matrix)
            acc, (prec, rec, f1, _) = accuracy_score(GT_true_qs, GT_pred_qs), precision_recall_fscore_support(GT_true_qs, GT_pred_qs, average="binary")
            critic_metric_dict["GT_acc"].append(acc); critic_metric_dict["GT_rec"].append(rec); critic_metric_dict["GT_prec"].append(prec); critic_metric_dict["GT_f1"].append(f1)

            # 1-others in prec_rec_f1 because sklearn wants true class as 1 and others has true class 0 (only for the sake of metric scores)
            acc, (prec, rec, f1, _) = accuracy_score(others_true_qs, others_pred_qs), precision_recall_fscore_support(1-others_true_qs, 1-others_pred_qs, average="binary") 
            critic_metric_dict["others_acc"].append(acc); critic_metric_dict["others_rec"].append(rec); critic_metric_dict["others_prec"].append(prec); critic_metric_dict["others_f1"].append(f1)

            mean_pred_qs = np.concatenate([GT_pred_qs, others_pred_qs], axis=0)
            mean_true_qs = np.concatenate([GT_true_qs, others_true_qs], axis=0)
            acc, (prec, rec, f1, _) = accuracy_score(mean_true_qs, mean_pred_qs), precision_recall_fscore_support(mean_true_qs, mean_pred_qs, average="binary")
            critic_metric_dict["mean_acc"].append(acc); critic_metric_dict["mean_rec"].append(rec); critic_metric_dict["mean_prec"].append(prec); critic_metric_dict["mean_f1"].append(f1)

            # Print
            metric_df = pd.DataFrame(columns=["GT_acc", "GT_rec", "GT_prec", "GT_f1", "others_acc", "others_rec", "others_prec", "others_f1", 
                                            "mean_acc", "mean_rec", "mean_prec", "mean_f1",  "time(epoch_start-now)"])

            critic_metric_dict["time(epoch_start-now)"].append(f"{(time.time()-start_time)/60:.2f} min")
            for col in metric_df.columns:
                metric_df[col] = [critic_metric_dict[col][-1]]
            metric_df.index = [epoch]
            print(tabulate(metric_df, headers='keys', tablefmt='fancy_grid'))
            print()

            

            # Update best model (with GT f1 - we want critic for best GT)
            metric_for_best_model = "GT_f1"
            curr_metric = critic_metric_dict[metric_for_best_model][-1]
            if curr_metric > best_metric:
                best_metric = curr_metric
                best_model = type(model)()
                best_model.load_state_dict(model.state_dict())
                best_epoch = epoch
                print(f"BEST MODEL UPDATED! BEST {metric_for_best_model} = {best_metric}")

        # Update embedding model and action_embeddings
        embedding_model.load_state_dict(model.GIN.state_dict())
        action_embeddings = get_action_dataset_embeddings(embedding_model, action_rsigs, action_psigs)
        action_embeddings_norm = torch.linalg.norm(action_embeddings, axis=1)

# save everything
folder = f"models/supervised/{args.model_type}/steps={args.steps}||actor_loss={args.actor_loss}||negative_selection={args.negative_selection}"
os.makedirs(folder, exist_ok = True)

if train_actor:
    metric_dict = actor_metric_dict
    
    # Save fig
    fig = plt.figure(figsize=(8, 8))
    for dist in filter(lambda x: "mean" in x, metric_dict.keys()):
        plt.plot(metric_dict[dist], label=dist)
    plt.title(f"Offline RL (steps={args.steps})")
    plt.xlabel("epoch")
    plt.ylabel("ranking")
    plt.legend()
    # fig.show() # COMMENT THIS IN FINAL CODE
    fig.savefig(os.path.join(folder, "plot.png"))
else:
    metric_dict = critic_metric_dict

torch.save(model, os.path.join(folder, "model.pth"))
pd.DataFrame.from_dict(metric_dict).to_csv(os.path.join(folder, "metrics.csv"))
json.dump({
    "steps(trajectory length)": args.steps,
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
    "epochs": epochs, 
    "batch_size": batch_size,
    "train_samples": train_idx.shape,
    "valid_samples": valid_idx.shape,
    "topk": topk,
    "best_epoch": best_epoch,
    "best_rank": best_rank
}, open(os.path.join(folder, "config.txt"), 'w'))
print("Saved model at", folder)