import sys
import os
import json
import argparse
import random
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch


from environment.graph_env import GraphEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def state_to_tensor(state, env):
    state_index = env.nodes.index(state)
    state_tensor = torch.zeros(env.num_nodes, device=device)
    state_tensor[state_index] = 1
    return state_tensor.unsqueeze(0)


def get_best_action(state, model, actions, env):
    """Select the action with the highest Q-value (greedy)."""
    with torch.no_grad():
        state_tensor = state_to_tensor(state, env)
        q_values = model(state_tensor)
        valid_indices = [env.nodes.index(a) for a in actions]
        valid_q_values = q_values[0, valid_indices]
        return actions[torch.argmax(valid_q_values).item()]


def rule_based_defender_action(env):
    """
    Mimics what a reasonable SOC analyst would do:

    1. Gather all nodes with status == 'Alerted'.
    2. Score each alerted node:
         score = unconditional_risk * 0.6
               + abs(z_score)        * 0.2
               + (1 if "CVE" in name else 0) * 0.2
    3. Pick the highest-scoring node as the target.
       - If the target name contains "CVE" -> action 2 (Restore Software / Patch)
       - Otherwise                        -> action 1 (Network Filtering)
    4. If no alerted nodes exist           -> action 0 (Do Nothing)

    Returns:
        (action_id, target_node_id or None)
    """
    alerted_nodes = []
    for node_id in env.node_list:
        node_data = env.graph.nodes[node_id]
        if node_data.get('status') == 'Alerted':
            alerted_nodes.append(node_id)

    if not alerted_nodes:
        return 0, None

    best_node = None
    best_score = -float('inf')

    for node_id in alerted_nodes:
        nd = env.graph.nodes[node_id]
        risk = float(nd.get('unconditional_risk', 0.0))
        z = float(nd.get('z_score', 0.0))
        name = nd.get('name', '')
        has_cve = 1.0 if "CVE" in name else 0.0

        score = risk * 0.6 + abs(z) * 0.2 + has_cve * 0.2

        if score > best_score:
            best_score = score
            best_node = node_id

    target_name = env.graph.nodes[best_node].get('name', '')
    if "CVE" in target_name:
        action_id = 2
    else:
        action_id = 1

    return action_id, best_node


ACTION_NAMES = {
    0: "Do Nothing",
    1: "Network Filtering",
    2: "Restore Software",
    3: "Restore Connection",
}


def run_episode(env, attacker_model, goal_node, max_steps):
    """
    Runs one full episode: attacker moves, then defender reacts.

    Returns a dict with per-episode metrics.
    """
    attacker_path = [env.current_node]
    defender_actions = []
    ep_reward = 0.0
    ep_won = False
    done = False
    outcome = "timeout"

    nodes_targeted = set()
    nodes_on_attack_path = set()

    for t in range(max_steps):
        attacker_current = env.current_node
        env.current_alert_group = []

        if attacker_current == goal_node:
            done = True
            ep_won = False
            outcome = "loss"
            break

        attacker_actions = env.get_valid_actions(attacker_current)

        if not attacker_actions:
            done = True
            ep_won = True
            outcome = "win"
            break

        next_move = get_best_action(attacker_current, attacker_model, attacker_actions, env)
        next_node, att_reward, alerts, step_done = env.step(next_move)
        env.current_node = next_node
        attacker_path.append(next_node)
        nodes_on_attack_path.add(next_node)

        if step_done:
            done = True
            if env.current_node == goal_node:
                ep_won = False
                outcome = "loss"
            else:
                ep_won = True
                outcome = "win"
            break

        action_id, target_node = rule_based_defender_action(env)
        defender_actions.append((action_id, target_node))

        response = env.apply_action(action_id, target_node=target_node)
        status = response.get('status', 'failed')

        if target_node is not None:
            nodes_targeted.add(target_node)

        env.compute_and_update_risk(decay_factor=0.2)

        if status == 'success' and action_id != 0:
            ep_reward += 1.0

    if nodes_targeted:
        hits = nodes_targeted & nodes_on_attack_path
        precision = len(hits) / len(nodes_targeted)
    else:
        precision = 0.0

    if nodes_on_attack_path:
        hits = nodes_targeted & nodes_on_attack_path
        recall = len(hits) / len(nodes_on_attack_path)
    else:
        recall = 0.0

    return {
        "outcome": outcome,
        "won": ep_won,
        "reward": ep_reward,
        "steps": len(defender_actions),
        "attacker_path": attacker_path,
        "defender_actions": defender_actions,
        "precision": precision,
        "recall": recall,
        "action_counts": _count_actions(defender_actions),
    }


def _count_actions(defender_actions):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for aid, _ in defender_actions:
        counts[aid] = counts.get(aid, 0) + 1
    return counts


def save_plots(results_df, output_dir):
    """Generate and save four evaluation plots."""

    outcome_counts = results_df['outcome'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['win', 'loss', 'timeout']
    values = [outcome_counts.get(c, 0) for c in categories]
    colors = ['#4CAF50', '#F44336', '#FF9800']
    ax.bar(categories, values, color=colors, edgecolor='black')
    ax.set_title('Episode Outcomes')
    ax.set_ylabel('Count')
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'outcome_distribution.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(results_df['reward'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(results_df['reward'].mean(), color='red', linestyle='--', label=f"Mean = {results_df['reward'].mean():.2f}")
    ax.set_title('Reward Distribution')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=150)
    plt.close(fig)

    total_actions = {0: 0, 1: 0, 2: 0, 3: 0}
    for counts_str in results_df['action_counts']:
        counts = counts_str if isinstance(counts_str, dict) else json.loads(counts_str.replace("'", '"'))
        for k, v in counts.items():
            total_actions[int(k)] += v

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [ACTION_NAMES[i] for i in sorted(total_actions.keys())]
    vals = [total_actions[i] for i in sorted(total_actions.keys())]
    bar_colors = ['#9E9E9E', '#2196F3', '#8BC34A', '#FF9800']
    ax.bar(labels, vals, color=bar_colors, edgecolor='black')
    ax.set_title('Defender Action Distribution (All Episodes)')
    ax.set_ylabel('Total Count')
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(results_df['recall'], results_df['precision'],
               alpha=0.4, s=18, c='teal', edgecolors='none')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (per episode)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(results_df['precision'].mean(), color='red', linestyle='--', alpha=0.6,
               label=f"Avg Prec = {results_df['precision'].mean():.2f}")
    ax.axvline(results_df['recall'].mean(), color='blue', linestyle='--', alpha=0.6,
               label=f"Avg Recall = {results_df['recall'].mean():.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Rule-Based Baseline Defender Evaluation")
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of evaluation episodes (default: 500)')
    parser.add_argument('--env_json', type=str, default='ag.json',
                        help='Path to the environment JSON file (default: ag.json)')
    parser.add_argument('--attacker_model', type=str,
                        default='policy-models/attacker/paper-hopes.h5',
                        help='Path to the pre-trained DQN attacker model')
    parser.add_argument('--goal_node', type=str, default='307',
                        help='Goal node ID for the attacker (default: 307)')
    parser.add_argument('--max_steps', type=int, default=10,
                        help='Maximum steps per episode (default: 10)')
    args = parser.parse_args()

    num_episodes = args.episodes
    max_steps = args.max_steps
    goal_node = args.goal_node

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    env_json_path = args.env_json if os.path.isabs(args.env_json) else os.path.join(project_root, args.env_json)
    attacker_model_path = (args.attacker_model if os.path.isabs(args.attacker_model)
                           else os.path.join(project_root, args.attacker_model))

    print(f"Loading environment from {env_json_path} ...")
    with open(env_json_path, 'r') as f:
        graph_json = json.load(f)

    env = GraphEnvironment(graph_json)
    env.end_node = goal_node
    print(f"  Nodes: {env.num_nodes}  |  Goal: {env.end_node}")

    print(f"Loading attacker model from {attacker_model_path} ...")
    attacker_model = torch.load(attacker_model_path, weights_only=False, map_location=device)
    attacker_model.eval()
    print("  Attacker model loaded.")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, 'baselines', 'results',
                              f'rule_based_defender_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    records = []
    wins = 0
    losses = 0
    timeouts = 0

    print(f"\nRunning {num_episodes} episodes (max {max_steps} steps each) ...\n")
    print(f"{'Ep':>6}  {'Outcome':<8}  {'Reward':>7}  {'Steps':>5}  {'Prec':>6}  {'Recall':>6}  {'Action':>8}")
    print("-" * 60)

    for ep in range(1, num_episodes + 1):
        env.reset_recon(ep)

        result = run_episode(env, attacker_model, goal_node, max_steps)

        if result['outcome'] == 'win':
            wins += 1
        elif result['outcome'] == 'loss':
            losses += 1
        else:
            timeouts += 1

        records.append({
            'episode': ep,
            'outcome': result['outcome'],
            'won': result['won'],
            'reward': result['reward'],
            'steps': result['steps'],
            'precision': result['precision'],
            'recall': result['recall'],
            'attacker_path': ' -> '.join(result['attacker_path']),
            'defender_actions': str([(ACTION_NAMES[a], t) for a, t in result['defender_actions']]),
            'action_counts': result['action_counts'],
        })

        if ep % 50 == 0 or ep == num_episodes:
            r = result
            act_summary = ', '.join(f"{ACTION_NAMES[k][0:3]}:{v}" for k, v in sorted(r['action_counts'].items()) if v > 0)
            print(f"{ep:>6}  {r['outcome']:<8}  {r['reward']:>7.2f}  {r['steps']:>5}  "
                  f"{r['precision']:>6.2f}  {r['recall']:>6.2f}  {act_summary}")

    results_df = pd.DataFrame(records)

    total = num_episodes
    win_rate = wins / total * 100
    loss_rate = losses / total * 100
    timeout_rate = timeouts / total * 100
    avg_reward = results_df['reward'].mean()
    std_reward = results_df['reward'].std()
    avg_precision = results_df['precision'].mean()
    avg_recall = results_df['recall'].mean()
    avg_steps = results_df['steps'].mean()

    summary = (
        f"\n{'=' * 60}\n"
        f"  RULE-BASED DEFENDER -- EVALUATION SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Episodes:         {total}\n"
        f"  Max steps/ep:     {max_steps}\n"
        f"  Goal node:        {goal_node}\n"
        f"{'=' * 60}\n"
        f"  Wins:             {wins:>5}  ({win_rate:.1f}%)\n"
        f"  Losses:           {losses:>5}  ({loss_rate:.1f}%)\n"
        f"  Timeouts:         {timeouts:>5}  ({timeout_rate:.1f}%)\n"
        f"{'=' * 60}\n"
        f"  Avg Reward:       {avg_reward:.3f}  (std {std_reward:.3f})\n"
        f"  Avg Precision:    {avg_precision:.3f}\n"
        f"  Avg Recall:       {avg_recall:.3f}\n"
        f"  Avg Steps:        {avg_steps:.2f}\n"
        f"{'=' * 60}\n"
    )
    print(summary)

    csv_path = os.path.join(output_dir, 'episode_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV saved to {csv_path}")

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Summary saved to {summary_path}")

    save_plots(results_df, output_dir)

    print(f"\nDone. All results in: {output_dir}\n")


if __name__ == '__main__':
    main()
