import os
import sys
import io
import json
import glob
import argparse
import contextlib
from collections import deque

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

_REAL_STDOUT = sys.stdout


def log(msg):
    print(msg, file=_REAL_STDOUT, flush=True)


@contextlib.contextmanager
def silenced():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


with silenced():
    from environment import graph_env as ge_mod
    from rewards.defender_reward import RewardModelPPO
    from agents.defender.ac_defender import GNN_ActorCritic, AC_Def_Agent
    from agents.defender.dqn_defender import GNN_QNetwork, DQN_Def_Agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_NAMES = ['Do Nothing', 'Network Filtering', 'Restore Software', 'Restore Connection']
MAX_STEPS = 10


def pick_goal_id(data):
    accesses = [int(k) for k, v in data['assets'].items()
                if v.get('metaconcept') == 'Access']
    if not accesses:
        return None
    return str(max(accesses))


def bfs_masked(env, src, tgt):
    if src == tgt:
        return [src]
    visited = {src}
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in env.get_valid_actions(u):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            if v == tgt:
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            q.append(v)
    return None


@torch.no_grad()
def select_ac(agent, x, ei, mask):
    import torch
    x_g = x.to(agent.device)
    ei_g = ei.to(agent.device)
    m_g = mask.to(agent.device)
    x_nodes, shared, batch_idx = agent.policy.encode(x_g, ei_g)
    node_dist = agent.policy.get_node_dist(x_nodes, shared, batch_idx, m_g)
    target = int(node_dist.logits.argmax().item())
    action_dist = agent.policy.get_action_dist(
        x_nodes, shared, torch.tensor([target], device=agent.device)
    )
    a_type = int(action_dist.logits.argmax(dim=-1).item())
    return a_type, target


@torch.no_grad()
def select_dqn(agent, x, ei, mask):
    return agent.act(x, ei, mask, epsilon=0.0)


def run_episode(env, model_kind, agent):
    goal_id = env.end_node
    env.reset_recon(0)

    entry = None
    for cand in env.candidates:
        if bfs_masked(env, cand, goal_id) is not None:
            entry = cand
            break
    if entry is None:
        return None

    env.current_node = entry
    env.previous_node = entry
    env.history = [entry]

    reward_model = RewardModelPPO()
    reward_model.total_reward = 25
    reward_model.stack = 0

    ep_reward = 0.0
    ep_won = False
    done = False
    attacker_path = [entry]
    defender_hits = set()
    defender_attempts = 0
    defender_effective = 0
    action_counts = [0] * 4

    for t in range(MAX_STEPS):
        if env.current_node == goal_id:
            done = True
            ep_won = False
            break

        path = bfs_masked(env, env.current_node, goal_id)
        if path is None or len(path) < 2:
            done = True
            ep_won = True
            outcome = "defense_success"
            terminal = reward_model.get_reward(
                env, 0, True, outcome, False, ep_won, 0,
                node=0, step=t, ep_precision=0.0,
            )
            ep_reward += terminal
            break

        next_node = path[1]
        nn, _, _, step_done = env.step(next_node)
        env.current_node = nn
        attacker_path.append(nn)

        if step_done:
            done = True
            ep_won = env.current_node != goal_id
            outcome = "defense_success" if ep_won else "compromise"
            terminal = reward_model.get_reward(
                env, 0, True, outcome, False, ep_won, 0,
                node=0, step=t, ep_precision=0.0,
            )
            ep_reward += terminal
            break

        x, ei = env.get_graph_observation()
        mask = env.get_valid_action_mask()

        if model_kind == 'ac':
            a_type, target_idx = select_ac(agent, x, ei, mask)
        else:
            a_type, target_idx = select_dqn(agent, x, ei, mask)

        target_id = env.node_list[target_idx]
        risk_before = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))
        target_risk = float(env.graph.nodes[target_id].get('unconditional_risk', 0.0))

        resp = env.apply_action(a_type, target_node=target_id)

        status = resp.get('status')
        changes = resp.get('changes', 0)
        action_counts[a_type] += 1

        if a_type != 0:
            defender_attempts += 1
            pair = (a_type, target_id)
            if status == 'success' and changes > 0 and pair not in defender_hits:
                defender_effective += 1
                defender_hits.add(pair)

        env.compute_and_update_risk(decay_factor=0.2)
        risk_after = float(env.graph.nodes[env.end_node].get('unconditional_risk', 0.0))

        step_reward = reward_model.get_reward(
            env, a_type, status == 'success', "normal", False, False, 0,
            target_idx, step=t, changes=changes,
            risk_before=risk_before, risk_after=risk_after,
            target_risk=target_risk, gamma=0.999,
        )
        ep_reward += step_reward

    attacker_set = set(attacker_path)
    blocking_hits = [(a, n) for (a, n) in defender_hits if a in (1, 2)]
    on_path = [h for h in blocking_hits if h[1] in attacker_set]
    precision = (len(on_path) / len(blocking_hits)) if blocking_hits else 0.0
    validity = (defender_effective / defender_attempts) if defender_attempts else 0.0

    return {
        'reward': float(ep_reward),
        'precision': float(precision),
        'validity': float(validity),
        'won': bool(ep_won),
        'episode_len': len(attacker_path),
        'action_counts': action_counts,
        'attacker_path_len': len(attacker_path),
        'goal_reached': env.current_node == goal_id,
    }


def load_ac(path):
    agent = AC_Def_Agent()
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    agent.policy.load_state_dict(state)
    agent.policy.eval()
    return agent


def load_dqn(path):
    agent = DQN_Def_Agent()
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    agent.q_net.load_state_dict(state)
    agent.q_net.eval()
    return agent


LOADERS = {
    'ac':  ('AC',  load_ac),
    'dqn': ('DQN', load_dqn),
}


def evaluate_model(model_key, model_path, variant_files, out_dir):
    name, loader = LOADERS[model_key]
    log(f"\n=== Evaluating {name} from {model_path} ===")
    if not os.path.exists(model_path):
        log(f"  [skip] checkpoint not found: {model_path}")
        return None

    with silenced():
        agent = loader(model_path)

    rows = []
    skipped = 0
    crashed = 0
    csv_path = os.path.join(out_dir, f"eval_{model_key}.csv")
    fcsv = open(csv_path, 'w')
    fcsv.write("variant,reward,precision,validity,won,goal_reached,len,"
               "do_nothing,filter,patch,restore\n")

    for i, vf in enumerate(variant_files):
        try:
            with open(vf) as f:
                data = json.load(f)
            goal = pick_goal_id(data)
            if goal is None:
                skipped += 1
                continue
            with silenced():
                env = ge_mod.GraphEnvironment(data, goal_node=goal)
                result = run_episode(env, model_key, agent)
        except Exception as e:
            crashed += 1
            if crashed <= 5:
                log(f"  [crash] {os.path.basename(vf)}: {type(e).__name__}: {e}")
            continue

        if result is None:
            skipped += 1
            continue

        rows.append(result)
        ac_str = ",".join(str(c) for c in result['action_counts'])
        fcsv.write(
            f"{os.path.basename(vf)},{result['reward']:.4f},"
            f"{result['precision']:.4f},{result['validity']:.4f},"
            f"{int(result['won'])},{int(result['goal_reached'])},"
            f"{result['episode_len']},{ac_str}\n"
        )

        if (i + 1) % 50 == 0:
            log(f"  ...{i+1}/{len(variant_files)} graphs done "
                f"(usable={len(rows)}, skipped={skipped}, crashed={crashed})")

    fcsv.close()

    if not rows:
        log(f"  [warn] no usable variants for {name}")
        return None

    rewards = np.array([r['reward'] for r in rows])
    precs = np.array([r['precision'] for r in rows])
    valids = np.array([r['validity'] for r in rows])
    wins = np.array([1 if r['won'] else 0 for r in rows])
    lens = np.array([r['episode_len'] for r in rows])
    action_totals = np.sum([r['action_counts'] for r in rows], axis=0)

    summary = {
        'model': name,
        'checkpoint': model_path,
        'graphs_eval': len(rows),
        'graphs_skipped': skipped,
        'graphs_crashed': crashed,
        'reward_mean': float(rewards.mean()),
        'reward_std': float(rewards.std()),
        'reward_median': float(np.median(rewards)),
        'precision_mean': float(precs.mean()),
        'precision_std': float(precs.std()),
        'validity_mean': float(valids.mean()),
        'win_rate': float(wins.mean()),
        'episode_len_mean': float(lens.mean()),
        'action_totals': action_totals.tolist(),
    }

    log(
        f"  → graphs={len(rows)} | win%={summary['win_rate']*100:5.1f} "
        f"| reward={summary['reward_mean']:7.2f}±{summary['reward_std']:.2f} "
        f"| prec={summary['precision_mean']*100:5.1f}% "
        f"| valid={summary['validity_mean']*100:5.1f}% "
        f"| len={summary['episode_len_mean']:.2f}"
    )
    log(f"  CSV saved: {csv_path}")
    return summary


def write_report(summaries, out_dir):
    md_path = os.path.join(out_dir, "evaluation_report.md")
    with open(md_path, 'w') as f:
        f.write("# Defender cross-graph evaluation\n\n")
        f.write("- Attacker: deterministic shortest path on the masked graph\n")
        f.write("- Goal: highest-id Access node per variant (auto-detected)\n")
        f.write("- Defender policy: greedy (argmax) from each loaded checkpoint\n")
        f.write(f"- Max episode length: {MAX_STEPS} steps\n\n")

        f.write("## Aggregate metrics\n\n")
        f.write("| Model | N graphs | Win rate | Reward (mean±std) | Precision | "
                "Validity | Avg len |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            if s is None:
                continue
            f.write(
                f"| {s['model']} | {s['graphs_eval']} "
                f"| {s['win_rate']*100:.1f}% "
                f"| {s['reward_mean']:.2f} ± {s['reward_std']:.2f} "
                f"| {s['precision_mean']*100:.1f}% "
                f"| {s['validity_mean']*100:.1f}% "
                f"| {s['episode_len_mean']:.2f} |\n"
            )

        f.write("\n## Action mix (totals across all evaluated graphs)\n\n")
        f.write("| Model | Do Nothing | Network Filter | Restore Software | Restore Conn |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for s in summaries:
            if s is None:
                continue
            ac = s['action_totals']
            tot = max(1, sum(ac))
            f.write(
                f"| {s['model']} "
                f"| {ac[0]} ({ac[0]*100/tot:.1f}%) "
                f"| {ac[1]} ({ac[1]*100/tot:.1f}%) "
                f"| {ac[2]} ({ac[2]*100/tot:.1f}%) "
                f"| {ac[3]} ({ac[3]*100/tot:.1f}%) |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `Win rate`: fraction of episodes where the deterministic attacker "
                "failed to reach the goal within the step budget.\n")
        f.write("- `Precision`: of unique blocking actions (Filter/Patch), the share "
                "that landed on a node actually visited by the attacker.\n")
        f.write("- `Validity`: of non-idle action attempts, the share that produced "
                "a structural change in the graph.\n")
    log(f"\nReport saved: {md_path}")


def main():
    p = argparse.ArgumentParser(
        description="Zero-shot evaluation of AC and DQN defenders across attack graph variants."
    )
    p.add_argument('--variants', default='attack_graphs/variants',
                   help='Directory holding variation_*.json files')
    p.add_argument('--ac-model', default='policy-models/defender/ac_defender.pth',
                   help='Actor-Critic checkpoint path')
    p.add_argument('--dqn-model', default='policy-models/defender/dqn_defender.pth',
                   help='DQN checkpoint path')
    p.add_argument('--limit', type=int, default=0,
                   help='Cap number of graphs (0 = all)')
    p.add_argument('--out', default='evaluation_out',
                   help='Output directory for CSVs and report')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    variant_files = sorted(glob.glob(os.path.join(args.variants, 'variation_*.json')))
    if args.limit:
        variant_files = variant_files[:args.limit]
    log(f"Found {len(variant_files)} attack graphs in {args.variants}")

    targets = [
        ('ac',  args.ac_model),
        ('dqn', args.dqn_model),
    ]

    summaries = []
    for key, path in targets:
        s = evaluate_model(key, path, variant_files, args.out)
        summaries.append(s)

    write_report(summaries, args.out)
    log("\nDone.")


if __name__ == '__main__':
    main()
