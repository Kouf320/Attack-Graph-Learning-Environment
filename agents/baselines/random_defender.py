import sys
import os


import argparse
import json
import random
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from environment.graph_env import GraphEnvironment
from rewards.defender_reward import RewardModelPPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def state_to_tensor(state, env):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor.unsqueeze(0)


def get_best_action(state, model, actions, env):
    """Selects the action with the highest Q-value (greedy)."""
    with torch.no_grad():
        state_tensor = state_to_tensor(state, env)
        q_values = model(state_tensor)
        valid_indices = [env.nodes.index(a) for a in actions]
        valid_q_values = q_values[0, valid_indices]
        return actions[torch.argmax(valid_q_values).item()]


ACTION_NAMES = {
    0: 'Do Nothing',
    1: 'Network Filtering',
    2: 'Restore Software',
    3: 'Restore Connection',
}


def random_defender_select(env):
    """
    Pick a random action type (0-3) and a random target node from the valid
    action mask.  If the mask is minimal (no real alerts), default to doing
    nothing.

    Returns
    -------
    action_type : int
    target_node_id : str
    """
    mask = env.get_valid_action_mask()
    valid_indices = torch.where(mask)[0].tolist()

    if len(valid_indices) == 0 or (len(valid_indices) == 1 and valid_indices[0] == 0):
        target_node_id = env.node_list[0]
        return 0, target_node_id

    action_type = random.randint(0, 3)
    target_idx = random.choice(valid_indices)
    target_node_id = env.node_list[target_idx]
    return action_type, target_node_id


def run_episode(env, attacker_model, reward_model, goal_node, max_steps):
    """
    Run one episode: attacker (DQN) vs. random defender.

    Returns a dict with episode-level metrics.
    """
    env.reset_recon(0)

    ep_reward = 0.0
    ep_won = False
    done = False
    attacker_path = [env.current_node]
    defender_actions_log = []

    for t in range(max_steps):
        attacker_current_node = env.current_node
        env.current_alert_group = []

        if attacker_current_node == goal_node:
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
                attacker_path.append(next_node)

                if step_done:
                    done = True
                    ep_won = env.current_node != goal_node

        if done:
            outcome = 'defense_success' if ep_won else 'compromise'
            reward = reward_model.get_reward(
                env, 0, True, outcome, False, ep_won, 0, node=0, step=t,
            )
            ep_reward += reward
            break

        action_type, target_node_id = random_defender_select(env)

        risk_before = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))
        target_risk = float(env.graph.nodes[target_node_id].get('unconditional_risk', 0.0))

        def_response = env.apply_action(action_type, target_node=target_node_id)
        status = def_response.get('status', 'failed')
        changes = def_response.get('changes', 0)

        env.compute_and_update_risk(decay_factor=0.2)
        risk_after = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))

        reward = reward_model.get_reward(
            env,
            action_type,
            status == 'success',
            'normal',
            False,
            ep_won,
            0,
            node=0,
            step=t,
            changes=changes,
            risk_before=risk_before,
            risk_after=risk_after,
            target_risk=target_risk,
            gamma=0.99,
        )
        ep_reward += reward

        defender_actions_log.append({
            'step': t,
            'action_type': action_type,
            'action_name': ACTION_NAMES[action_type],
            'target_node': target_node_id,
            'status': status,
        })

    return {
        'won': ep_won,
        'reward': ep_reward,
        'steps': len(attacker_path),
        'attacker_path': attacker_path,
        'defender_actions': defender_actions_log,
    }


def evaluate(env, attacker_model, reward_model, goal_node, num_episodes, max_steps):
    results = []

    for ep in range(1, num_episodes + 1):
        ep_result = run_episode(env, attacker_model, reward_model, goal_node, max_steps)
        results.append(ep_result)

        if ep % 50 == 0 or ep == num_episodes:
            wins_so_far = sum(1 for r in results if r['won'])
            print(
                f"[{ep:>5}/{num_episodes}]  "
                f"Win rate: {wins_so_far / ep * 100:5.1f}%  |  "
                f"Avg reward: {np.mean([r['reward'] for r in results]):+.2f}"
            )

    return results


def build_dataframe(results):
    rows = []
    for i, r in enumerate(results):
        action_counts = Counter(a['action_type'] for a in r['defender_actions'])
        nodes_targeted = list(set(a['target_node'] for a in r['defender_actions']))

        rows.append({
            'episode': i + 1,
            'won': r['won'],
            'reward': r['reward'],
            'steps': r['steps'],
            'attacker_path': ' -> '.join(str(n) for n in r['attacker_path']),
            'num_defender_actions': len(r['defender_actions']),
            'do_nothing_count': action_counts.get(0, 0),
            'network_filter_count': action_counts.get(1, 0),
            'restore_sw_count': action_counts.get(2, 0),
            'restore_conn_count': action_counts.get(3, 0),
            'nodes_targeted': ', '.join(str(n) for n in nodes_targeted),
        })
    return pd.DataFrame(rows)


def print_summary(df):
    total = len(df)
    wins = df['won'].sum()
    losses = total - wins

    print("\n" + "=" * 60)
    print("  RANDOM DEFENDER BASELINE  --  SUMMARY")
    print("=" * 60)
    print(f"  Episodes           : {total}")
    print(f"  Wins               : {wins}  ({wins / total * 100:.1f}%)")
    print(f"  Losses             : {losses}  ({losses / total * 100:.1f}%)")
    print(f"  Avg reward         : {df['reward'].mean():+.2f}")
    print(f"  Std reward         : {df['reward'].std():.2f}")
    print(f"  Median reward      : {df['reward'].median():+.2f}")
    print(f"  Avg episode steps  : {df['steps'].mean():.1f}")
    print("-" * 60)
    print("  Action distribution (total across all episodes):")
    print(f"    Do Nothing         : {df['do_nothing_count'].sum()}")
    print(f"    Network Filtering  : {df['network_filter_count'].sum()}")
    print(f"    Restore Software   : {df['restore_sw_count'].sum()}")
    print(f"    Restore Connection : {df['restore_conn_count'].sum()}")
    print("=" * 60 + "\n")


def save_plots(df, output_dir):
    """Generate and save evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    wins = df['won'].sum()
    losses = len(df) - wins
    bars = ax.bar(['Win', 'Loss'], [wins, losses], color=['#2ecc71', '#e74c3c'])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontweight='bold',
        )
    ax.set_title('Random Defender: Win vs Loss')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'win_loss_bar.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df['reward'], bins=30, color='#3498db', edgecolor='white', alpha=0.85)
    ax.axvline(df['reward'].mean(), color='red', linestyle='--', label=f"Mean = {df['reward'].mean():.2f}")
    ax.set_title('Reward Distribution')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    action_labels = list(ACTION_NAMES.values())
    action_totals = [
        df['do_nothing_count'].sum(),
        df['network_filter_count'].sum(),
        df['restore_sw_count'].sum(),
        df['restore_conn_count'].sum(),
    ]
    colors = ['#95a5a6', '#e67e22', '#2ecc71', '#3498db']
    bars = ax.bar(action_labels, action_totals, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontweight='bold',
        )
    ax.set_title('Defender Action Distribution (Total)')
    ax.set_ylabel('Count')
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Random Defender Baseline evaluation'
    )
    parser.add_argument(
        '--episodes', type=int, default=500,
        help='Number of evaluation episodes (default: 500)',
    )
    parser.add_argument(
        '--env_json', type=str, default='ag.json',
        help='Path to the environment JSON file (default: ag.json)',
    )
    parser.add_argument(
        '--attacker_model', type=str,
        default='policy-models/attacker/paper-hopes.h5',
        help='Path to the DQN attacker model (default: policy-models/attacker/paper-hopes.h5)',
    )
    parser.add_argument(
        '--goal_node', type=str, default='307',
        help='Goal node ID for the attacker (default: 307)',
    )
    parser.add_argument(
        '--max_steps', type=int, default=10,
        help='Max steps per episode (default: 10)',
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    env_json_path = args.env_json
    if not os.path.isabs(env_json_path):
        env_json_path = os.path.join(project_root, env_json_path)

    attacker_model_path = args.attacker_model
    if not os.path.isabs(attacker_model_path):
        attacker_model_path = os.path.join(project_root, attacker_model_path)

    print(f"Loading environment from {env_json_path} ...")
    with open(env_json_path, 'r') as f:
        json_data = json.load(f)
    env = GraphEnvironment(json_data)

    print(f"Loading attacker model from {attacker_model_path} ...")
    attacker_model = torch.load(attacker_model_path, weights_only=False, map_location=device)
    attacker_model.eval()

    reward_model = RewardModelPPO()

    print(f"\nRunning {args.episodes} episodes  |  max_steps={args.max_steps}  |  goal={args.goal_node}")
    print("-" * 60)
    results = evaluate(
        env, attacker_model, reward_model,
        goal_node=args.goal_node,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    df = build_dataframe(results)
    print_summary(df)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(
        os.path.dirname(__file__), 'results', f'random_defender_{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'episode_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Per-episode CSV saved to {csv_path}")

    save_plots(df, output_dir)

    print("Done.")


if __name__ == '__main__':
    main()
