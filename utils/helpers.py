import re
from collections import defaultdict
import networkx as nx
import numpy as np

def _get_ip_from_node_name(node_name: str):
    """Helper function to extract an IP address from a node name string."""
    match = re.search(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)', node_name)
    return match.group(1) if match else 'no_ip'


def define_network_state(
    severity: float,
    node_bc: float,
    current_network_state: int
):

    network_state_mapping = {
            'Normal': 0,
            'Alerted': 1,
            'Attacked': 2
    }


    new_network_state = ''

    severity = 0.90
    
    if severity != 0:
        severity = severity * 5
        if severity >= 3.00 or node_bc > 0.10 or (severity >= 2.00 and node_bc > 0.10):
            new_network_state = network_state_mapping['Alerted']

        is_alerted = (current_network_state == network_state_mapping['Alerted'])
        if severity >= 4.80 or (severity >= 4.00 and is_alerted) or (severity >= 3.00 and node_bc > 0.10 and is_alerted):
            new_network_state = network_state_mapping['Attacked']
            new_timestep = 0

        return new_network_state, severity

    else:
        new_timestep += 1
        return new_network_state, 0

def update_attack_path_with_uncertainty(graph, committed_path, potential_forks, new_goal_nodes, 
                                          resolution_threshold=5, fork_ttl=50):
    """
    Updates the attack path using confidence scoring, a resolution window, and a fork "Time To Live" (TTL).
    *** NEW: This version avoids creating new forks to IPs already in the committed path or other forks. ***
    
    Fork object is: {'path': List[Nodes], 'score': int, 'age': int}
    """
    if not new_goal_nodes:
        for fork_obj in potential_forks:
            fork_obj['age'] += 1
        
        pruned_forks = [f for f in potential_forks if f['age'] <= fork_ttl]
        return committed_path, pruned_forks

    if not committed_path and not potential_forks:
        committed_path.append(new_goal_nodes[0])
        print(f"🚀 Attack initiated. Starting at: {graph.nodes[committed_path[-1]]['name']}")
        return committed_path, []

    last_committed_node = committed_path[-1]

    if potential_forks:
        print(f"🔎 Updating {len(potential_forks)} forks...")
        new_goal_ips = {_get_ip_from_node_name(graph.nodes[goal]['name']) for goal in new_goal_nodes}
        existing_fork_ips = set()
        
        for fork_obj in potential_forks:
            fork_ip = _get_ip_from_node_name(graph.nodes[fork_obj['path'][-1]]['name'])
            existing_fork_ips.add(fork_ip)
            
            if fork_ip in new_goal_ips:
                fork_obj['score'] += 1
                fork_obj['age'] = 0
                print(f"   Score +1 for fork to {fork_ip} (Score: {fork_obj['score']}, Age: 0)")
            else:
                fork_obj['age'] += 1
                print(f"   Aging fork to {fork_ip} (Score: {fork_obj['score']}, Age: {fork_obj['age']})")

        committed_path_ips = {_get_ip_from_node_name(graph.nodes[node]['name']) for node in committed_path}
        forbidden_ips = existing_fork_ips.union(committed_path_ips)

        new_fork_ips_to_create = new_goal_ips - forbidden_ips
        
        for ip in new_fork_ips_to_create:
            paths_to_new_ip = []
            goals_for_ip = [g for g in new_goal_nodes if _get_ip_from_node_name(graph.nodes[g]['name']) == ip]
            
            for goal in goals_for_ip:
                if nx.has_path(graph, last_committed_node, goal):
                    path_segment = nx.shortest_path(graph, last_committed_node, goal)[1:]
                    if path_segment:
                        paths_to_new_ip.append(path_segment)
            
            if paths_to_new_ip:
                shortest_new_fork_path = min(paths_to_new_ip, key=len)
                new_fork_obj = {'path': shortest_new_fork_path, 'score': 1, 'age': 0}
                potential_forks.append(new_fork_obj)
                print(f"🚦 New target IP {ip}. Creating new fork with score 1, age 0.")

        surviving_forks = [f for f in potential_forks if f['age'] <= fork_ttl]
        if len(potential_forks) != len(surviving_forks):
            print(f"   Pruned {len(potential_forks) - len(surviving_forks)} stale forks (older than {fork_ttl} steps).")
        
        if not surviving_forks:
            return committed_path, []

        sorted_forks = sorted(surviving_forks, key=lambda f: f['score'], reverse=True)
        
        top_fork = sorted_forks[0]
        winner = None

        if len(sorted_forks) == 1:
            if top_fork['score'] >= resolution_threshold:
                winner = top_fork
        else:
            second_fork = sorted_forks[1]
            if (top_fork['score'] - second_fork['score']) >= resolution_threshold:
                winner = top_fork
        
        if winner:
            winner_ip = _get_ip_from_node_name(graph.nodes[winner['path'][-1]]['name'])
            print(f"✅ EUREKA! Fork to {winner_ip} won with score {winner['score']} (Threshold: {resolution_threshold}). Committing path.")
            committed_path.extend(winner['path'])
            return update_attack_path_with_uncertainty(graph, committed_path, [], new_goal_nodes, resolution_threshold, fork_ttl)
        else:
            print(f"   No fork met resolution threshold. Uncertainty remains.")
            return committed_path, surviving_forks

    else:
        print("🚦 Path is certain. Looking for new forks...")
        paths_by_ip = defaultdict(list)
        for goal in new_goal_nodes:
            if nx.has_path(graph, last_committed_node, goal):
                path_segment = nx.shortest_path(graph, last_committed_node, goal)[1:]
                if path_segment:
                    destination_ip = _get_ip_from_node_name(graph.nodes[path_segment[-1]]['name'])
                    paths_by_ip[destination_ip].append(path_segment)
        
        if not paths_by_ip:
            print("   New goals not reachable from committed path.")
            return committed_path, []
            
        committed_path_ips = {_get_ip_from_node_name(graph.nodes[node]['name']) for node in committed_path}
        print(f"   Detected activity on {len(paths_by_ip)} new IPs. Filtering against committed path...")

        for ip, path_list in paths_by_ip.items():
            if ip not in committed_path_ips:
                new_fork_obj = {
                    'path': min(path_list, key=len),
                    'score': 1,
                    'age': 0
                }
                potential_forks.append(new_fork_obj)
                print(f"   Creating new fork for {ip}.")
            else:
                print(f"   Ignoring fork to {ip} (already in committed path).")

        return committed_path, potential_forks
    

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count