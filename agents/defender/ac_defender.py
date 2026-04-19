import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
import sys
import os
import random
from datetime import datetime
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import softmax as geo_softmax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.colors import BColors
from rewards.defender_reward import RewardModelPPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def state_to_tensor(state, env):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor.unsqueeze(0)


def get_best_action(state, model, actions, env):
    with torch.no_grad():
        state_tensor = state_to_tensor(state, env)
        q_values = model(state_tensor)
        valid_indices = [env.nodes.index(a) for a in actions]
        valid_q_values = q_values[0, valid_indices]
        return actions[torch.argmax(valid_q_values).item()]


class GNN_ActorCritic(nn.Module):
    """
    Shared GATConv encoder with separate actor (node + type) and critic heads.

    Differs from the PPO agent:
      - No LSTM (stateless — each step is independent)
      - Same factored action space: sample node → sample action type
    """

    def __init__(self, node_feature_dim, action_type_dim=4, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        num_heads = 4
        head_dim = hidden_dim // num_heads

        self.conv1 = GATConv(node_feature_dim, head_dim, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)
        self.bn2 = BatchNorm(hidden_dim)

        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)

        context_dim = hidden_dim * 2

        self.actor_node_score = nn.Linear(context_dim, 1)
        self.actor_type = nn.Linear(context_dim, action_type_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_node_score.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_type.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def encode(self, x, edge_index, batch_index=None):
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        x_nodes = F.relu(self.bn2(self.conv2(h, edge_index)))

        global_feat = global_mean_pool(x_nodes, batch_index)
        shared = F.relu(self.shared_fc(global_feat))

        return x_nodes, shared, batch_index

    def get_node_dist(self, x_nodes, shared, batch_index, node_mask=None):
        global_expanded = shared[batch_index]
        node_ctx = torch.cat([x_nodes, global_expanded], dim=1)
        node_scores = self.actor_node_score(node_ctx).squeeze(-1)

        if node_mask is not None:
            node_scores = node_scores.masked_fill(~node_mask.bool(), -1e9)

        return Categorical(logits=node_scores)

    def get_action_dist(self, x_nodes, shared, target_node_idx):
        selected = x_nodes[target_node_idx]
        if selected.dim() == 1:
            selected = selected.unsqueeze(0)
        if shared.dim() == 1:
            shared = shared.unsqueeze(0)
        combined = torch.cat([shared, selected], dim=1)
        logits = self.actor_type(combined)
        return Categorical(logits=logits)

    def get_value(self, shared):
        return self.critic(shared)


class EpisodeBuffer:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.is_terminals = []

    def push(self, log_prob, value, reward, entropy, is_terminal=False):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.is_terminals.append(bool(is_terminal))

    def clear(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.is_terminals = []

    def __len__(self):
        return len(self.rewards)


class AC_Def_Agent:
    def __init__(self):
        self.device = device
        self.reward_model = RewardModelPPO()

        self.action_mapping = {
            'Do Nothing': 0, 'Network Filtering': 1,
            'Restore Software': 2, 'Restore Connection': 3,
        }
        self.action_type_dim = len(self.action_mapping)
        self.node_feat_dim = 17

        self.lr = 3e-4
        self.gamma = 0.99
        self.entropy_coeff = 0.01
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5
        self.hidden_dim = 256
        self.update_every = 1

        self.goals = ['307']

        self.policy = GNN_ActorCritic(self.node_feat_dim, self.action_type_dim,
                                      self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
        )

        self.buffer = EpisodeBuffer()


    def act(self, x, edge_index, node_mask):
        """
        Sample action from the policy.

        Returns (action_type, target_node_idx, log_prob, value, entropy).
        """
        x_gpu = x.to(self.device)
        ei_gpu = edge_index.to(self.device)
        mask_gpu = node_mask.to(self.device)

        x_nodes, shared, batch_idx = self.policy.encode(x_gpu, ei_gpu)

        node_dist = self.policy.get_node_dist(x_nodes, shared, batch_idx, mask_gpu)
        target_node = node_dist.sample()

        action_dist = self.policy.get_action_dist(x_nodes, shared, target_node.unsqueeze(0))
        action_type = action_dist.sample()

        log_prob = (node_dist.log_prob(target_node) + action_dist.log_prob(action_type)).squeeze()

        value = self.policy.get_value(shared).squeeze()

        entropy = action_dist.entropy().squeeze()

        return (action_type.item(), target_node.item(),
                log_prob, value, entropy)


    def update(self):
        """
        Standard A2C update with MC returns.
        """
        if len(self.buffer) == 0:
            return {}

        returns = []
        G = 0.0
        for r, term in zip(reversed(self.buffer.rewards),
                           reversed(self.buffer.is_terminals)):
            if term:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        log_probs = torch.stack(self.buffer.log_probs)
        values = torch.stack(self.buffer.values)
        entropies = torch.stack(self.buffer.entropies)

        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages).mean()

        value_loss = F.mse_loss(values, returns)

        entropy_loss = -self.entropy_coeff * entropies.mean()

        loss = policy_loss + self.value_coeff * value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.clear()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropies.mean().item(),
        }


    def train_agent(self, env, num_episodes, output_path, attacker_model_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_params = (f"AC_lr{self.lr}_g{self.gamma}_ent{self.entropy_coeff}"
                      f"_hdim{self.hidden_dim}")
        run_folder = f"runs/{run_timestamp}_{run_params}"
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "training_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("Episode,WinRate,AvgReward,AvgLen,ValidityRatio,"
                    "Entropy,PolicyLoss,ValueLoss,AvgPrecision,AvgTimeToBlock\n")

        attacker_model = torch.load(attacker_model_path, weights_only=False, map_location=device)
        attacker_model.eval()

        all_rewards = []
        recent_wins = []
        recent_lengths = []
        recent_validity = []
        all_ep_precision = []
        all_ep_goal_risk = []
        all_ep_action_counts = []
        all_ep_time_to_block = []
        all_ep_outcomes = []
        all_ep_attacker_paths = []
        all_ep_defender_actions = []
        recent_update_metrics = []

        action_counts = {k: 0 for k in self.action_mapping.keys()}
        best_reward = -999999

        last_entropy = 0.0
        last_pol_loss = 0.0
        last_val_loss = 0.0

        print(f"{BColors.HEADER}{'Ep':<6} | {'Win%':<5} | {'Rwd':<6} | {'Len':<5} | "
              f"{'Valid%':<6} | {'Prec%':<6} | {'Entr':<5} | {'Action Dist':<20}{BColors.ENDC}")
        print("-" * 90)

        for episode in range(1, num_episodes + 1):
            env.current_alert_group = ''
            env.reset_recon(episode)
            ep_reward = 0.0
            ep_len = 0
            ep_won = False
            done = False

            self.reward_model.total_reward = 25
            self.reward_model.stack = 0

            ep_valid_actions = 0
            ep_non_idle_attempts = 0
            ep_effective_actions = 0

            ep_attacker_path = [env.current_node]
            ep_defender_actions = []
            ep_defender_unique_hits = set()
            ep_goal_risk_trajectory = []
            ep_first_block_step = -1
            ep_action_type_counts = {k: 0 for k in self.action_mapping.keys()}

            a_target_idx = 0

            for t in range(10):
                ep_len += 1

                attacker_current_node = env.current_node
                env.current_alert_group = []

                if attacker_current_node in self.goals:
                    done = True
                    ep_won = False
                else:
                    attacker_actions = env.get_valid_actions(attacker_current_node)
                    if not attacker_actions:
                        done = True
                        ep_won = True
                    else:
                        next_move = get_best_action(
                            attacker_current_node, attacker_model, attacker_actions, env
                        )
                        next_node, att_r, alerts, step_done = env.step(next_move)
                        env.current_node = next_node
                        ep_attacker_path.append(next_node)

                        if step_done:
                            done = True
                            ep_won = env.current_node not in self.goals

                if done:
                    outcome = "defense_success" if ep_won else "compromise"
                    terminal_reward = self.reward_model.get_reward(
                        env, 0, True, outcome, False, ep_won, episode,
                        node=a_target_idx, step=t,
                    )
                    ep_reward += terminal_reward

                    self.buffer.push(
                        torch.tensor(0.0, device=self.device),
                        torch.tensor(0.0, device=self.device),
                        terminal_reward,
                        torch.tensor(0.0, device=self.device),
                        is_terminal=True,
                    )
                    break

                x, edge_index = env.get_graph_observation()
                mask = env.get_valid_action_mask()

                a_type, a_target_idx, log_prob, value, entropy = self.act(
                    x, edge_index, mask
                )

                target_node_id = env.node_list[a_target_idx]
                target_node_data = env.graph.nodes[target_node_id]

                risk_before = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))
                target_risk = float(target_node_data.get('unconditional_risk', 0.0))

                def_response = env.apply_action(a_type, target_node=target_node_id)
                status = def_response.get('status')
                changes = def_response.get('changes', 0)

                if status == 'success':
                    ep_valid_actions += 1
                    if a_type != 0 and ep_first_block_step == -1:
                        ep_first_block_step = t

                if a_type != 0:
                    ep_non_idle_attempts += 1
                    pair = (a_type, target_node_id)
                    if status == 'success' and changes > 0 and pair not in ep_defender_unique_hits:
                        ep_effective_actions += 1
                        ep_defender_unique_hits.add(pair)

                ep_defender_actions.append((a_type, target_node_id))
                ep_action_type_counts[list(self.action_mapping.keys())[a_type]] += 1

                env.compute_and_update_risk(decay_factor=0.2)
                risk_after = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))
                ep_goal_risk_trajectory.append(risk_after)

                step_reward = self.reward_model.get_reward(
                    env, a_type, status == 'success', "normal", False, ep_won, episode,
                    a_target_idx, step=t, changes=changes,
                    risk_before=risk_before, risk_after=risk_after,
                    target_risk=target_risk, gamma=self.gamma,
                )
                ep_reward += step_reward
                action_counts[list(self.action_mapping.keys())[a_type]] += 1

                self.buffer.push(log_prob, value, step_reward, entropy,
                                 is_terminal=False)

            if episode % self.update_every == 0:
                metrics = self.update()
                recent_update_metrics.append(metrics)
                self.scheduler.step()

            all_rewards.append(ep_reward)
            recent_wins.append(1 if ep_won else 0)
            recent_lengths.append(ep_len)

            if ep_non_idle_attempts > 0:
                recent_validity.append(ep_effective_actions / ep_non_idle_attempts)
            else:
                recent_validity.append(0.0)

            attack_path_set = set(ep_attacker_path)
            defensive_hits = [(a, tgt) for a, tgt in ep_defender_unique_hits if a in (1, 2)]
            if defensive_hits:
                on_path = sum(1 for _, tgt in defensive_hits if tgt in attack_path_set)
                ep_precision = on_path / len(defensive_hits)
            else:
                ep_precision = 0.0

            all_ep_attacker_paths.append(ep_attacker_path)
            all_ep_defender_actions.append(ep_defender_actions)
            all_ep_precision.append(ep_precision)
            all_ep_goal_risk.append(ep_goal_risk_trajectory)
            all_ep_action_counts.append(ep_action_type_counts)
            all_ep_time_to_block.append(ep_first_block_step)
            outcome = 'win' if ep_won else ('loss' if done else 'timeout')
            all_ep_outcomes.append(outcome)

            if episode % 100 == 0:
                avg_rew = np.mean(all_rewards[-100:])
                win_rate = np.mean(recent_wins[-100:]) * 100
                avg_len = np.mean(recent_lengths[-100:])
                avg_valid = np.mean(recent_validity[-100:]) * 100
                avg_prec = np.mean(all_ep_precision[-100:]) * 100
                valid_ttb = [x for x in all_ep_time_to_block[-100:] if x >= 0]
                avg_ttb = np.mean(valid_ttb) if valid_ttb else -1

                if recent_update_metrics:
                    avg_pol_loss = np.mean([m.get('policy_loss', 0) for m in recent_update_metrics])
                    avg_val_loss = np.mean([m.get('value_loss', 0) for m in recent_update_metrics])
                    avg_entropy = np.mean([m.get('entropy', 0) for m in recent_update_metrics])
                    last_pol_loss = avg_pol_loss
                    last_val_loss = avg_val_loss
                    last_entropy = avg_entropy
                else:
                    avg_pol_loss = last_pol_loss
                    avg_val_loss = last_val_loss
                    avg_entropy = last_entropy

                act_dist = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                act_str = (f"{act_dist[0][0][0:3]}:{act_dist[0][1]}, "
                           f"{act_dist[1][0][0:3]}:{act_dist[1][1]}" if len(act_dist) >= 2 else "")

                print(f"{'-'*90}")
                print(f"{episode:<6} | {win_rate:>5.1f} | {avg_rew:>6.1f} | {avg_len:>5.1f} | "
                      f"{avg_valid:>6.1f} | {avg_prec:>6.1f} | {avg_entropy:>5.2f} | {act_str}")
                print(f"{'-'*90}")

                with open(csv_path, "a") as f:
                    f.write(f"{episode},{win_rate},{avg_rew},{avg_len},{avg_valid},"
                            f"{avg_entropy},{avg_pol_loss},{avg_val_loss},"
                            f"{avg_prec},{avg_ttb}\n")

                self.generate_charts(csv_path, run_folder)
                action_counts = {k: 0 for k in self.action_mapping.keys()}
                recent_update_metrics = []

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(self.policy.state_dict(), output_path)

        self.generate_final_visualisations(
            run_folder=run_folder,
            all_rewards=all_rewards,
            recent_wins=recent_wins,
            all_ep_precision=all_ep_precision,
            all_ep_goal_risk=all_ep_goal_risk,
            all_ep_action_counts=all_ep_action_counts,
            all_ep_time_to_block=all_ep_time_to_block,
            all_ep_outcomes=all_ep_outcomes,
            all_ep_attacker_paths=all_ep_attacker_paths,
            all_ep_defender_actions=all_ep_defender_actions,
            csv_path=csv_path,
            env=env,
            num_episodes=num_episodes,
        )


    def generate_charts(self, csv_path, run_folder="logs"):
        try:
            df = pd.read_csv(csv_path)
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            ax1 = axs[0, 0]
            ax1.plot(df['Episode'], df['WinRate'], label='Win %', color='green')
            ax1.plot(df['Episode'], df['ValidityRatio'], label='Valid Action %', color='blue', linestyle='--')
            if 'AvgPrecision' in df.columns:
                ax1.plot(df['Episode'], df['AvgPrecision'], label='Precision %', color='red', linestyle=':')
            ax1.set_title("Performance Quality")
            ax1.legend()

            ax2 = axs[0, 1]
            ax2.plot(df['Episode'], df['AvgReward'], color='orange')
            ax2.set_title("Average Reward")

            ax3 = axs[1, 0]
            ax3.plot(df['Episode'], df['AvgLen'], label='Ep Length', color='purple')
            ax3.set_title("Episode Length")

            ax4 = axs[1, 1]
            ax4.plot(df['Episode'], df['Entropy'], color='gray')
            ax4.set_title("Policy Entropy")

            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "training_dashboard.png"))
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def generate_final_visualisations(self, run_folder, all_rewards, recent_wins,
                                       all_ep_precision, all_ep_goal_risk,
                                       all_ep_action_counts, all_ep_time_to_block,
                                       all_ep_outcomes, all_ep_attacker_paths,
                                       all_ep_defender_actions, csv_path, env,
                                       num_episodes):
        print(f"\n{'='*60}")
        print(f"Generating final visualisations in: {run_folder}")
        print(f"{'='*60}")

        window = min(100, max(10, num_episodes // 20))

        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

            win_arr = np.array([1 if w else 0 for w in recent_wins], dtype=float)
            if len(win_arr) >= window:
                win_smooth = np.convolve(win_arr, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(win_arr)), win_smooth * 100,
                             color='green', linewidth=1.5)
            else:
                axes[0].plot(win_arr * 100, color='green', alpha=0.5)
            axes[0].set_ylabel('Win Rate (%)')
            axes[0].set_title('A2C Training Curves')
            axes[0].grid(True, alpha=0.3)

            rew_arr = np.array(all_rewards, dtype=float)
            if len(rew_arr) >= window:
                rew_smooth = np.convolve(rew_arr, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(rew_arr)), rew_smooth,
                             color='orange', linewidth=1.5)
            axes[1].plot(rew_arr, color='orange', alpha=0.15)
            axes[1].set_ylabel('Episode Reward')
            axes[1].grid(True, alpha=0.3)

            prec_arr = np.array(all_ep_precision, dtype=float)
            if len(prec_arr) >= window:
                prec_smooth = np.convolve(prec_arr, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window-1, len(prec_arr)), prec_smooth * 100,
                             color='red', linewidth=1.5)
            axes[2].plot(prec_arr * 100, color='red', alpha=0.15)
            axes[2].set_ylabel('Action Precision (%)')
            axes[2].set_xlabel('Episode')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig1_training_curves.png"), dpi=200)
            plt.close()
            print("  [OK] fig1_training_curves.png")

            fig, ax = plt.subplots(figsize=(14, 6))
            action_names = list(self.action_mapping.keys())
            chunk_size = max(1, num_episodes // 20)
            x_ticks = []
            action_fracs = {name: [] for name in action_names}

            for i in range(0, num_episodes, chunk_size):
                chunk = all_ep_action_counts[i:i+chunk_size]
                if not chunk:
                    continue
                total = sum(sum(ep.values()) for ep in chunk)
                if total == 0:
                    total = 1
                x_ticks.append(i + chunk_size // 2)
                for name in action_names:
                    count = sum(ep.get(name, 0) for ep in chunk)
                    action_fracs[name].append(count / total)

            bottom = np.zeros(len(x_ticks))
            colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71']
            for j, name in enumerate(action_names):
                vals = np.array(action_fracs[name])
                ax.bar(x_ticks, vals, bottom=bottom, width=chunk_size * 0.8,
                       label=name, color=colors[j % len(colors)])
                bottom += vals

            ax.set_xlabel('Episode')
            ax.set_ylabel('Action Fraction')
            ax.set_title('A2C Action Distribution Evolution')
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig2_action_distribution_evolution.png"), dpi=200)
            plt.close()
            print("  [OK] fig2_action_distribution_evolution.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            max_steps = 10

            win_risks = [r for r, o in zip(all_ep_goal_risk, all_ep_outcomes)
                         if o == 'win' and len(r) > 0]
            loss_risks = [r for r, o in zip(all_ep_goal_risk, all_ep_outcomes)
                          if o == 'loss' and len(r) > 0]

            def _avg_trajectory(trajs, max_len):
                padded = []
                for tr in trajs:
                    p = tr + [tr[-1]] * (max_len - len(tr)) if len(tr) < max_len else tr[:max_len]
                    padded.append(p)
                return np.mean(padded, axis=0) if padded else np.zeros(max_len)

            if win_risks:
                avg_win = _avg_trajectory(win_risks, max_steps)
                ax.plot(range(max_steps), avg_win, color='green', linewidth=2,
                        label=f'Win (n={len(win_risks)})')
            if loss_risks:
                avg_loss = _avg_trajectory(loss_risks, max_steps)
                ax.plot(range(max_steps), avg_loss, color='red', linewidth=2,
                        label=f'Loss (n={len(loss_risks)})')

            ax.set_xlabel('Defender Step')
            ax.set_ylabel('Goal Node Unconditional Risk')
            ax.set_title('Goal Node Risk Trajectory (Win vs Loss)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig3_risk_reduction_trajectory.png"), dpi=200)
            plt.close()
            print("  [OK] fig3_risk_reduction_trajectory.png")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            outcome_counts = Counter(all_ep_outcomes)
            labels = list(outcome_counts.keys())
            sizes = list(outcome_counts.values())
            colors_pie = {'win': '#2ecc71', 'loss': '#e74c3c', 'timeout': '#f39c12'}
            pie_colors = [colors_pie.get(l, '#95a5a6') for l in labels]

            axes[0].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Episode Outcome Distribution')

            total_actions = Counter()
            for ep in all_ep_action_counts:
                for k, v in ep.items():
                    total_actions[k] += v
            ax_names = list(total_actions.keys())
            ax_vals = list(total_actions.values())
            bars = axes[1].bar(ax_names, ax_vals, color=colors[:len(ax_names)])
            axes[1].set_title('Overall Action Type Distribution')
            axes[1].set_ylabel('Count')
            for bar in bars:
                h = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                             ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig4_outcome_and_action_distribution.png"), dpi=200)
            plt.close()
            print("  [OK] fig4_outcome_and_action_distribution.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            valid_ttb = [x for x in all_ep_time_to_block if x >= 0]
            if valid_ttb:
                ax.hist(valid_ttb, bins=range(0, max_steps + 2), color='#3498db',
                        edgecolor='white', alpha=0.8, rwidth=0.85)
                ax.axvline(np.mean(valid_ttb), color='red', linestyle='--',
                           label=f'Mean: {np.mean(valid_ttb):.1f}')
                ax.legend()
            ax.set_xlabel('Step of First Successful Block')
            ax.set_ylabel('Frequency')
            ax.set_title('Time-to-Block Distribution')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig5_time_to_block.png"), dpi=200)
            plt.close()
            print("  [OK] fig5_time_to_block.png")

            fig, ax = plt.subplots(figsize=(6, 5))
            tp, fp, fn = 0, 0, 0
            for path, actions, outcome in zip(all_ep_attacker_paths,
                                               all_ep_defender_actions, all_ep_outcomes):
                path_set = set(path)
                for a_type, tgt in actions:
                    if a_type == 0:
                        continue
                    if tgt in path_set:
                        tp += 1
                    else:
                        fp += 1
                acted_targets = set(tgt for at, tgt in actions if at != 0)
                for node in path_set:
                    if node not in acted_targets:
                        fn += 1

            matrix = np.array([[tp, fp], [fn, 0]])
            ax.imshow(matrix, cmap='Blues', aspect='auto')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                            fontsize=14, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['On Attack Path', 'Off Attack Path'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Acted On', 'Not Acted On'])
            ax.set_title('A2C Defense Action Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "fig6_confusion_matrix.png"), dpi=200)
            plt.close()
            print("  [OK] fig6_confusion_matrix.png")

            try:
                fig, ax = plt.subplots(figsize=(16, 12))
                G = env.graph
                pos = nx.spring_layout(G, seed=42, k=0.15, iterations=30)

                node_action_freq = Counter()
                for ep_actions in all_ep_defender_actions:
                    for a_type, tgt in ep_actions:
                        if a_type != 0:
                            node_action_freq[tgt] += 1

                risk_vals = [env.graph.nodes[n].get('unconditional_risk', 0.0) for n in G.nodes()]
                max_freq = max(node_action_freq.values()) if node_action_freq else 1
                node_sizes = [300 + 700 * (node_action_freq.get(n, 0) / max_freq) for n in G.nodes()]

                nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                                      node_color=risk_vals, cmap=plt.cm.YlOrRd,
                                                      vmin=0, vmax=1, alpha=0.85, ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=[env.end_node], node_size=800,
                                       node_color='none', edgecolors='blue', linewidths=3, ax=ax)

                top_targets = [n for n, _ in node_action_freq.most_common(10)]
                label_map = {n: env.graph.nodes[n].get('name', n)[:20] for n in top_targets}
                nx.draw_networkx_labels(G, pos, labels=label_map, font_size=7, ax=ax)

                plt.colorbar(nodes_drawn, ax=ax, label='Unconditional Risk')
                ax.set_title('A2C Attack Graph Heatmap\n(Node size = action frequency, Color = risk)')
                plt.tight_layout()
                plt.savefig(os.path.join(run_folder, "fig7_attack_graph_heatmap.png"), dpi=200)
                plt.close()
                print("  [OK] fig7_attack_graph_heatmap.png")
            except Exception as e:
                print(f"  [SKIP] fig7_attack_graph_heatmap.png: {e}")

            ep_rows = []
            for i in range(num_episodes):
                ep_rows.append({
                    'episode': i + 1,
                    'outcome': all_ep_outcomes[i],
                    'reward': all_rewards[i],
                    'precision': all_ep_precision[i],
                    'time_to_block': all_ep_time_to_block[i],
                    'attacker_path': ' -> '.join(str(n) for n in all_ep_attacker_paths[i]),
                })
            pd.DataFrame(ep_rows).to_csv(
                os.path.join(run_folder, "episode_details.csv"), index=False
            )
            print("  [OK] episode_details.csv")

        except Exception as e:
            print(f"Final visualisation error: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Actor-Critic Defender Training')
    parser.add_argument('--episodes', type=int, default=16000)
    parser.add_argument('--env_json', type=str, default='ag.json')
    parser.add_argument('--attacker_model', type=str,
                        default='policy-models/attacker/paper-hopes.h5')
    parser.add_argument('--goal_node', type=str, default='307')
    parser.add_argument('--output', type=str,
                        default='policy-models/def-ac/ac_defender.h5')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    env_json_path = args.env_json
    if not os.path.isabs(env_json_path):
        env_json_path = os.path.join(project_root, env_json_path)

    attacker_path = args.attacker_model
    if not os.path.isabs(attacker_path):
        attacker_path = os.path.join(project_root, attacker_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    print(f"Loading environment from {env_json_path} ...")
    with open(env_json_path, 'r') as f:
        json_data = json.load(f)

    from environment.graph_env import GraphEnvironment
    environment = GraphEnvironment(json_data)

    agent = AC_Def_Agent()
    agent.train_agent(environment, args.episodes, output_path, attacker_path)


if __name__ == '__main__':
    main()
