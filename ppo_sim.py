# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import random
import time
import os
from distutils.util import strtobool

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from offlineRL_utils import *
from action_utils import *
from utils import calc_reward

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=-1, help="Cude device. -1 is cpu")
    parser.add_argument("--processes", type=int, default=10, help="Cude device. -1 is cpu")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--batch-size", type=int, default=2048, help="the number of parallel game environments")
    parser.add_argument("--minibatch-size", type=int, default=64, help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()
    args.num_steps = 5
    return args


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.GIN = torch.load("models/zinc2m_gin.pth")
        self._actor = NeuralNet(256, 256, num_hidden=3, hidden_size=500)
        self.actor_log_std = nn.Parameter(torch.zeros(self._actor.last_layer.out_features, dtype=torch.float32)).to(device)
        self._critic = NeuralNet(256, 1, num_hidden=2, hidden_size=256)
    
    def actor(self, x):
        return self.forward(x[0], x[1], out_type="actor")
    
    def critic(self, x):
        return self.forward(x[0], x[1], out_type="critic")

    def _pack_mols(self, mol_list):
        molecules = list(map(lambda x: data.Molecule.from_molecule(x, atom_feature="pretrain", bond_feature="pretrain"), mol_list))
        return data.Molecule.pack(molecules).to(device)

    def forward(self, reac, prod, out_type="both"):
        '''
        If out_type="actor", returns actions
        If out_type="critic", returns q_value
        If out_type="both", returns [actions, q_value]
        '''
        reac = self._pack_mols(reac)
        prod = self._pack_mols(prod)
        reac_out = self.GIN(reac, reac.node_feature.float())["graph_feature"]
        prod_out = self.GIN(prod, prod.node_feature.float())["graph_feature"]
        inp = torch.concat([reac_out, prod_out], axis=1)
    
        output = []
        if out_type in ["both", "actor"]:
            output.append(self._actor(inp))

        if out_type in ["both", "critic"]:
            output.append(self._critic(inp))
        
        if len(output) == 1:
            return output[0]
        return output

    def get_action_and_value(self, x, actions=None):
        actions = self.actor(x)
        action_logstd = self.actor_log_std.expand_as(actions)
        probs = Normal(actions, action_logstd.exp())
        if actions is None:
            actions = probs.sample()
        return actions, probs.log_prob(actions).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_value(self, x):
        return self.critic(x)

def generate_target(mol):
    for i in range(5):
        actions = get_applicable_actions(mol)
        if actions.shape[0] == 0:
            return mol
        try:
            mol = apply_action(mol, *actions.sample().iloc[0])
        except:
            pass
    return mol

def alternate_apply_action(args):
    mol, action = args
    if action is None:
        return mol
    try:
        mol = apply_action(mol, *action)
    except:
        pass
    return mol

class ENV:
    def __init__(self, agent_embedder= None, batch_size=32, processes=10) -> None:
        self.start_mols = pd.read_pickle("datasets/my_uspto/unique_start_mols.pickle")
        self.all_targets = pd.read_pickle("models/onlineRL_sim/targets.pickle")
        self.batch_size = batch_size
        self.processes = processes

        # Molecule embedder
        self.GIN = agent_embedder

        # action related things
        # action_dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)
        # action_dataset = action_dataset.loc[action_dataset["action_tested"] & action_dataset["action_works"]]
        # action_dataset = action_dataset[["rsub", "rcen", "rsig", "rbond", "psub", "pcen", "psig", "pbond"]]

        # self.action_rsigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["rsig"]))).to(device)
        # self.action_psigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["psig"]))).to(device)
        
        self.timestep = 0
        self.last_emb_update = -10e6
        self.reset()

    def _update_embeddings(self):
        self.action_embeddings = torch.tensor(np.load("models/onlineRL_sim/init_act_emb.npy")).to(device) # FIXME
        # self.action_embeddings = get_action_dataset_embeddings(self.GIN, 
        #                                                 self.action_rsigs,
        #                                                 self.action_psigs,
        #                                                 128 if args.cuda==-1 else 2048)

    def reset(self):
        self.cur_mols = list(map(Chem.MolFromSmiles, self.start_mols.sample(self.batch_size).values))
        self.targets = list(map(Chem.MolFromSmiles, self.all_targets.sample(self.batch_size).values))

        # Need to update 50 times
        if self.timestep - self.last_emb_update > 1e5:
            print(f"Updating embeddings. Last updated = {self.last_emb_update}. Current timestep = {self.timestep}")
            self._update_embeddings()
            self.last_emb_update = self.timestep

        return (self.cur_mols, self.targets)
    
    def step(self, actions):
        with Pool(self.processes) as P:
            app_act_list = P.map(get_applicable_actions, self.cur_mols, chunksize=chunksize)

        act_to_apply_list = []
        for i in range(len(self.cur_mols)):
            temp_act_emb = self.action_embeddings[dataset.index.isin(app_act_list[i].index)] 
            if temp_act_emb.shape[0] == 0: # Do nothing if no action applicable
                act_to_apply_list.append(None)
                continue
            temp_dist = torch.linalg.norm(temp_act_emb - actions[i], axis=1)
            temp_act = app_act_list[i].iloc[temp_dist.argmin().item()]
            act_to_apply_list.append(temp_act)
        
        with Pool(self.processes) as P:
            next_mols = P.map(alternate_apply_action, zip(self.cur_mols, act_to_apply_list), chunksize=chunksize)

        rewards = []
        for i in range(len(self.cur_mols)):
            rewards.append(
                calc_reward(
                    self.cur_mols[i],
                    None,
                    next_mols[i],
                    self.targets[i],
                    'sim'
                )
            )
        rewards = torch.Tensor(rewards).to(device)
        self.cur_mols = next_mols

        done = torch.Tensor([1 if act is None else 0 for act in act_to_apply_list]).to(device)

        # Update timestep -- needed for updating action embeddings
        self.timestep += len(self.cur_mols)

        return (self.cur_mols, self.targets), rewards, done, None

if __name__ == "__main__":
    chunksize = 10
    args = parse_args()
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cpu" if args.cuda == -1 or not torch.cuda.is_available() else f"cuda:{args.cuda}")

    agent = ActorCritic().to(device)
    env = ENV(agent_embedder=agent.GIN, batch_size=args.batch_size, processes=args.processes)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    actions = torch.zeros((5, args.batch_size, 256)).to(device)
    logprobs = torch.zeros((5, args.batch_size)).to(device)
    rewards = torch.zeros((5, args.batch_size)).to(device)
    dones = torch.zeros((5, args.batch_size)).to(device)
    values = torch.zeros((5, args.batch_size)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_done = torch.zeros(args.batch_size).to(device)
    num_updates = args.total_timesteps // args.batch_size

    metrics_to_save = {
        "rewards": [],
        "returns": [],
        "v_loss": [],
        "pg_loss": [],
        "exp_var": [],
    }
    for update in tqdm.tqdm(range(1, num_updates + 1)):
        obs_trainable = []
        tt = time.time()
        next_obs = env.reset()
        print(f"Took {time.time() - tt} s for env.reset()")
        for step in range(0, args.num_steps):
            global_step += 1 * args.batch_size
            obs_trainable.append(next_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                tt = time.time()
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                print(f"Took {time.time() - tt} s for agent.get_action()")
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            tt = time.time()
            next_obs, reward, done, infos = env.step(action)
            print(f"Took {time.time() - tt} s for env.step()")
            # print(f"{done.sum()} / {done.shape}")
            rewards[step] = reward.detach().clone().to(device).view(-1)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # obs = torch.stack(obs_trainable, axis=0)
        b_obs = (
            sum([item[0] for item in obs_trainable], []), # mol on i'th step
            sum([item[1] for item in obs_trainable], []), # Target
        )#obs.reshape(5*args.batch_size, -1)#.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(5*args.batch_size, -1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value((b_obs[0][start:end], b_obs[1][start:end]), b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics_to_save["rewards"].append(rewards.mean().item())
        metrics_to_save["returns"].append(returns.mean().item())
        metrics_to_save["v_loss"].append(v_loss.item())
        metrics_to_save["pg_loss"].append(pg_loss.item())
        metrics_to_save["exp_var"].append(explained_var)


        if update % 10 == 0:
            os.makedirs("models/onlineRL_sim", exist_ok=True)
            torch.save(agent, "models/onlineRL_sim/model.pth")
            pickle.dump(metrics_to_save, open("models/onlineRL_sim/rewards.pickle", 'wb'))
