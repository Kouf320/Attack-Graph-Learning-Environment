import json
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
#import pygraphviz as pgv
import networkx as nx
from pyvis.network import Network
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import requests
import torch
import sqlite3
from torch_geometric.utils import from_networkx
import numpy as np
from collections import deque, defaultdict
from utils.Alert import AlertGenerator
from utils.Colors import BColors
import re
import logging
from rewards.attacker.RewardModel import RewardModel, DefaultRewardModel
from collections import Counter


def load_config(path='config.json'):
    """Loads the JSON configuration file."""
    with open(path, 'r') as f:
        return json.load(f)


class GraphEnvironment:

    def __init__(self, json):
        logging.basicConfig(filename="logs/env_logs.log",
                            encoding="utf-8",
                            filemode="a+",
                            format="{asctime} - {levelname} - {message}",
                            style="{",
                            datefmt="%Y-%m-%d %H:%M",
                            level=logging.INFO)

        self.reward_model = DefaultRewardModel()
        config = load_config()
        env_config = config['environment_settings']
        self.NORM_F = env_config['NORM_F']['value']
        self.NORM_F_ATTEMPT = env_config['NORM_F_ATTEMPT']['value']
        self.NON_ACTION_VALUE = env_config['NON_ACTION_VALUE']['value']
        self.NVD_KEY = env_config['NVD_KEY']['value']
        self.DATABASE_PATH = env_config['DATABASE_PATH']['value']
        
        self.defensive_action_history = []
        self.removed_edges = []
        
        self.json_graph = json
        self.alert_gen = AlertGenerator()

        self.ip_context = defaultdict(lambda: {
            'sum_severities': 0.0,
            'sum_sq_severities': 0.0,
            'total_alerts': 0,
            'severity_counts': defaultdict(int)
        })
        self.ip_to_nodes_map = defaultdict(list)

        self.vulnerability_scores = {}
        self.alert_history = deque(maxlen=50)
        self.centrality = {}
        self.history = []
        self.vuln_met = 0
        self.total_vuln = 0

        self.graph, _, self.name_to_id_map = self.initialize_from_json() 

        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)

        self._map_ips_to_nodes()

        self.base_adjacency = nx.adjacency_matrix(self.graph, nodelist=self.node_list).todense().astype(float)

        self.node_mask = np.ones(self.num_nodes, dtype=int)
        self.edge_mask = np.ones((self.num_nodes, self.num_nodes), dtype=int)

        self.adjacency_matrix = self._compute_masked_adjacency()
        
        self._update_graph_state()

        self.candidates = []
        self.current_candidate = ''
        
        for node, data in self.graph.nodes(data=True):
            name = data.get('name', '')
            if name.startswith("Reconnaissance"): 
                self.candidates.append(node)
        
        if not self.candidates:
            print("Warning: No Reconnaissance nodes found in graph. Defaulting to random.")
            self.candidates = list(self.graph.nodes())
            if self.end_node in self.candidates:
                self.candidates.remove(self.end_node)

        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)

        #self.end_node = self.nodes[-1]
        self.end_node = '307'

        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        self.end_label = self.graph.nodes[str(self.end_node)]['name']
        match = re.search(ip_pattern, self.end_label)

        self.goal_ip = match.group() if match else None

        #self.end_node = self.nodes[-3]
        self.current_alert = ""
        self.noise_prob = 0.00 #0.05
        self.realistic_alerts = 1
        self.last_host_ip = "13.12.4.20"

        self.volume = 30
        self.offset_a = 20

        self.current_alert_group = []

        #This probabably needs to change with the graph model that comes as an inout
        self.current_node = 0
        self.previous_node = 0
        self.current_host = 0
        self.previous_host = -1

        # PRAYERS FOR CURICULUM Training

        self.curriculum_map = {}
        
        self.calc_nearest_nodes()

        self.compute_and_update_risk(decay_factor=0.2)
        
    def _update_graph_state(self):
        print("...Updating graph state (nodes, adjacency matrix)...")
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        self.adjacency_matrix = nx.adjacency_matrix(self.graph)

    def _update_alert_volume(self, alert_groups):
        batch_total = len(alert_groups)
        if batch_total == 0:
            return {}

        self.total_alert_count += batch_total

        batch_src_counts = Counter()
        batch_dst_counts = Counter()

        for alert in alert_groups:
            batch_src_counts[alert.get('src_ip')] += 1
            batch_dst_counts[alert.get('dest_ip')] += 1
        
        batch_ips = set(batch_src_counts) | set(batch_dst_counts)

        for ip in batch_ips:
            if ip not in self.ip_to_nodes_map:
                continue

            for node_id in self.ip_to_nodes_map[ip]:
                node = self.graph.nodes[node_id]
                
                if 'cum_src_count' not in node: node['cum_src_count'] = 0
                if 'cum_dst_count' not in node: node['cum_dst_count'] = 0
                
                node['cum_src_count'] += batch_src_counts[ip]
                node['cum_dst_count'] += batch_dst_counts[ip]

        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            
            s_count = node.get('cum_src_count', 0)
            d_count = node.get('cum_dst_count', 0)
            
            if self.total_alert_count > 0:
                node['alert_volume_src'] = s_count / self.total_alert_count
                node['alert_volume_dst'] = d_count / self.total_alert_count
            else:
                node['alert_volume_src'] = 0.0
                node['alert_volume_dst'] = 0.0

        return {}

    def _update_ip_metrics(self, ip_address, new_severities):
        if not ip_address or not new_severities:
            return

        ctx = self.ip_context[ip_address]
        
        batch_mean = np.mean(new_severities)
        
        for sev in new_severities:
            ctx['severity_counts'][sev] += 1
            ctx['total_alerts'] += 1
            ctx['sum_severities'] += sev
            ctx['sum_sq_severities'] += (sev ** 2)

        entropy = 0.0
        if ctx['total_alerts'] > 0:
            for count in ctx['severity_counts'].values():
                if count > 0:
                    p = count / ctx['total_alerts']
                    entropy -= p * np.log2(p)

        z_score = 0.0
        if ctx['total_alerts'] > 1:
            global_mean = ctx['sum_severities'] / ctx['total_alerts']
            global_variance = (ctx['sum_sq_severities'] / ctx['total_alerts']) - (global_mean ** 2)
            
            if global_variance < 0: global_variance = 0 
            global_std = np.sqrt(global_variance)
            
            if global_std > 1e-6:
                z_score = (batch_mean - global_mean) / global_std

        for node_id in self.ip_to_nodes_map[ip_address]:
            self.graph.nodes[node_id]['z_score'] = float(z_score)
            self.graph.nodes[node_id]['entropy'] = float(entropy)

    def _map_ips_to_nodes(self):
        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        for node in self.graph.nodes():
            name = self.graph.nodes[node].get('name', '')
            match = re.search(ip_pattern, name)
            if match:
                ip = match.group()
                self.ip_to_nodes_map[ip].append(node)
                
            self.graph.nodes[node]['z_score'] = 0.0
            self.graph.nodes[node]['entropy'] = 0.0

    def reset_defender(self):
        self.reset_mask() 
        self.current_node = self.nodes[0] 
        
        node_probs = self.compute_and_update_risk()
        
        risk_state = [node_probs.get(n, 0.0) for n in self.nodes]
        
        return np.array(risk_state, dtype=np.float32)

    def reset(self): #REQUIRED FOR EVERY NEW INSTANCE
        #self.current_node = random.choice(self.nodes)
        self.current_host = 0
        self.previous_host = -1
        self.total_alert_count = 0
        self.previous_node = 0
        self.current_node = self.nodes[0]
        self.current_alert_group = []

        self.defensive_action_history = [] 
        
        self.reset_mask()

        self.vuln_met = 0

        return self.current_node

    def random_reset(self): # RANDOM RESET IS BROKEN FIX BEFORE START CODE
        #self.current_node = random.choice(self.nodes)
        self.current_host = 0
        self.previous_host = -1
        self.total_alert_count = 0
        self.previous_node = 0

        self.current_alert_group = []

        self.current_node = random.choice(list(self.graph.nodes()))
        
        self.current_host, node_of_c = self.find_node_with_name_from_predecessors(self.current_node, "Reconnaisance")

        predecessors = list(self.graph.predecessors(self.current_node))

        if predecessors:
            self.previous_node = random.choice(predecessors)
        else:
            self.previous_node = 0

        self.reset_mask()

        self.vuln_met = 0

        return self.current_node
    
    def reset_curriculum(self, episode_num):
        candidates = []
        self.total_alert_count = 0
        
        hops_allowed = []
        
        if episode_num < 1000:
            hops_allowed = [1]          # Baby steps: Only immediate neighbors of goal
        elif episode_num < 2000:
            hops_allowed = [1, 2]       # Easy: 2 steps away
        elif episode_num < 3000:
            hops_allowed = [1, 2, 3]    # Medium
        elif episode_num < 4000:
            hops_allowed = [1, 2, 3, 4] 
        elif episode_num < 5000:
            hops_allowed = [1, 2, 3, 4, 5, 6, 7] 
        else:
            hops_allowed = list(self.curriculum_map.keys())

        for h in hops_allowed:
            candidates.extend(self.curriculum_map.get(h, []))

        if not candidates:
            candidates = list(self.graph.nodes())
            if self.end_node in candidates:
                candidates.remove(self.end_node)
        
        start_node = random.choice(candidates)

        if (episode_num == 0 or episode_num == 1000 or 
            episode_num == 2000 or 
            episode_num == 3000 or 
            episode_num == 4000 or 
            episode_num == 5000):
            
            print(candidates)

        self.reset_mask() 
        self.current_alert_group = []
        self.vuln_met = 0
        self.total_vuln = 0 
        
        self.current_node = start_node
        self.current_host, _ = self.find_node_with_name_from_predecessors(self.current_node, "Reconnaisance")
        
        predecessors = list(self.graph.predecessors(self.current_node))
        if predecessors:
            self.previous_node = random.choice(predecessors)
        else:
            self.previous_node = 0 
            
        if self.previous_node != 0:
            prev_node_name = self.graph.nodes[str(self.previous_node)]['name']
            match = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", prev_node_name)
            self.previous_host = match.group() if match else -1
        else:
            self.previous_host = -1

        return self.current_node
    
    def reset_recon(self, episode):
        # if episode % 2000 == 0:
        #     print("Candidates are now -> " + str(self.candidates))
        #     print("Candidates lenght is now -> " + str(len(self.candidates)))
        #     if episode != 0:
        #         self.candidates.remove(self.current_candidate)
        #     self.current_candidate = random.choice(self.candidates)
        #     print("Episode: " + str(episode) + "   Changed Current Candidate to -> " + self.current_candidate)
        
        self.total_alert_count = 0

        self.reset_mask() 
        self.current_alert_group = []
        self.vuln_met = 0
        self.total_vuln = 0 
        
        #self.current_node = self.current_candidate
        self.current_node = random.choice(self.candidates)
        
        self.current_host, _ = self.find_node_with_name_from_predecessors(self.current_node, "Recon")
        
        predecessors = list(self.graph.predecessors(self.current_node))

        if predecessors:
            self.previous_node = random.choice(predecessors)
        else:
            self.previous_node = 0 
            
        if self.previous_node != 0:
            prev_node_name = self.graph.nodes[str(self.previous_node)]['name']
            match = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", prev_node_name)
            self.previous_host = match.group() if match else -1
        else:
            self.previous_host = -1

        return self.current_node
    
    def calc_nearest_nodes(self):
        try:
            print(f"Pre-computing curriculum map for Goal: {self.end_node}...")
            
            path_lengths = nx.single_target_shortest_path_length(self.graph, self.end_node)
            
            for node, hops in path_lengths.items():
                if node == self.end_node: continue 
                
                if hops not in self.curriculum_map:
                    self.curriculum_map[hops] = []
                self.curriculum_map[hops].append(node)
                
            max_hops = max(self.curriculum_map.keys()) if self.curriculum_map else 0
            print(f"Curriculum Map Ready. Max distance: {max_hops} hops.")
            
        except Exception as e:
            print(f"Warning: Curriculum pre-computation failed: {e}")
            self.curriculum_map = {}

    def set_current_node(self, node):
        self.current_node = node
        return self.current_node
    
    def step(self, action):
        next_node = action
        curr_idx = self.node_to_idx.get(self.current_node)
        next_idx = self.node_to_idx.get(next_node)
        
        is_valid_move = False
        if curr_idx is not None and next_idx is not None:
            if self.node_mask[curr_idx] == 1 and self.node_mask[next_idx] == 1:
                if self.base_adjacency[curr_idx, next_idx] > 0 and self.edge_mask[curr_idx, next_idx] == 1:
                    is_valid_move = True

        self.current_alert = ""
        
        if not is_valid_move:
            return self.current_node, -1000, self.current_alert, True  
        
        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"

        source_ip_for_alert = self.last_host_ip 

        curr_node_name = self.graph.nodes[str(self.current_node)]['name']
        
        match_prev = re.search(ip_pattern, curr_node_name)
        if match_prev:
            self.last_host_ip = match_prev.group()

        distance = self.graph[self.current_node][next_node]['distance']
        reward = distance
        self.current_alert_group = []

        if self.reward_model.is_vulnerability_met(self.graph, self.current_node):
            self.vuln_met += 1

        self.previous_node = self.current_node
        self.current_node = next_node
        self.history.append(self.current_node)

        current_node_data = self.graph.nodes[str(self.current_node)]
        current_node_name = current_node_data.get('name', '')
       
        alerts_by_ip = defaultdict(list)
        base_prob = current_node_data.get('detection_prob', 0.0)
        base_sev = current_node_data.get('base_severity', 4)
        alert_type = current_node_data.get('alert_type', 'unknown')

        range_num = 1
        if self.realistic_alerts == 1:        
            range_num = random.randint(self.volume-self.offset_a, self.volume+self.offset_a)

        match_dest = re.search(ip_pattern, current_node_name)
        dest_ip_for_alert = match_dest.group() if match_dest else source_ip_for_alert

        # Failsafes
        # if self.previous_node == 0: source_ip_for_alert = "13.12.4.20" 
        # if not dest_ip_for_alert: dest_ip_for_alert = source_ip_for_alert
        # if not source_ip_for_alert: source_ip_for_alert = "13.12.4.20"
        
        self.previous_host = source_ip_for_alert
        self.current_host = dest_ip_for_alert

        # print("OH BOY ---------------------")
        # print("prev = " + str(self.previous_host))
        # print("curr = " + str(self.current_host))
        # print("-----------------------------")

        random_triggers = []
        # RANDOM NOISE
        #print(self.graph.nodes)

        for n in self.graph.nodes:
            n_name = self.graph.nodes[n].get('name', '')
            ip_match = re.search(ip_pattern, n_name)
            if ip_match:
                random_triggers.append(ip_match.group())
        
        target_count = max(1, int(len(random_triggers) * 0.05))
        
        if random_triggers:
            targets = random.sample(random_triggers, min(target_count, len(random_triggers)))
            # print(" RANDOM NODES SELECTED ------->", targets)
            for target_ip in targets:
                alert = self.alert_gen.generate_alert_recon(
                    source_ip_for_alert, random.randint(1024, 65535), 
                    target_ip, 80, 
                    severity=base_sev
                )
                self.current_alert_group.append(alert)

        for item in range(1, range_num):
            
            if random.random() < base_prob:
                if alert_type == 'recon':
                    neighbors = list(self.graph.neighbors(self.current_node))
                    
                    connected_ips = []
                    for n in neighbors:
                        n_name = self.graph.nodes[n].get('name', '')
                        ip_match = re.search(ip_pattern, n_name)
                        if ip_match:
                            connected_ips.append(ip_match.group())
                    
                    target_count = max(1, int(len(connected_ips) * 0.05))
                    
                    if connected_ips:
                        targets = random.sample(connected_ips, min(target_count, len(connected_ips)))
                        # print(" NEIGHBORS SELECTED ------->", targets)
                        for target_ip in targets:
                            alert = self.alert_gen.generate_alert_recon(
                                source_ip_for_alert, random.randint(1024, 65535), 
                                target_ip, 80, 
                                severity=base_sev
                            )
                            self.current_alert_group.append(alert)
                    else:
                        alert = self.alert_gen.generate_alert_recon(
                            source_ip_for_alert, random.randint(1024, 65535), 
                            dest_ip_for_alert, 80, 
                            severity=base_sev
                        )
                        self.current_alert_group.append(alert)

                elif alert_type == 'local':
                    self.current_alert = self.alert_gen.generate_alert_local(
                        dest_ip_for_alert, random.randint(1024, 65535), 
                        dest_ip_for_alert, random.randint(1024, 65535),
                        severity=base_sev
                    )
                    self.current_alert_group.append(self.current_alert)

                elif alert_type == 'network':
                    if "CVE" in current_node_name:
                        edge_data = self.graph[self.previous_node][self.current_node]
                        cvss_based_value = edge_data.get('distance', -1)
                        if cvss_based_value > 8: base_sev = 5
                        elif cvss_based_value > 6: base_sev = 4
                    
                    self.current_alert = self.alert_gen.generate_alert_network(
                        source_ip_for_alert, random.randint(1024, 65535), 
                        dest_ip_for_alert, 445,
                        severity=base_sev
                    )
                    self.current_alert_group.append(self.current_alert)

            # Noise generation
            if not self.current_alert_group and random.random() < self.noise_prob:
                noise = self.alert_gen.generate_alert_noise()
                self.current_alert_group.append(noise)

        for node in self.graph.nodes():
            if self.graph.nodes[node].get('status') != 'Compromised':
                self.graph.nodes[node]['status'] = 'Normal'

        ip_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
        alerted_ips = set()
        
        for alert_obj in self.current_alert_group:
            alert_str = str(alert_obj) 
            found_ips = ip_pattern.findall(alert_str)
            for ip in found_ips:
                alerted_ips.add(ip)
        
        for ip in alerted_ips:
            if ip in self.ip_to_nodes_map:
                for node_id in self.ip_to_nodes_map[ip]:
                    self.graph.nodes[node_id]['status'] = 'Alerted'
                    # print(f"DEBUG: Node {node_id} ({ip}) set to ALERTED")

        alert_group_t = self.current_alert_group
        if self.current_alert_group:
            for alert_obj in self.current_alert_group:
                severity = alert_obj.get('alert', {}).get('severity', 1)
                src_ip = self.previous_host 
                dst_ip = self.current_host  
                
                if src_ip: alerts_by_ip[src_ip].append(severity)
                if dst_ip: alerts_by_ip[dst_ip].append(severity)

        self._update_alert_volume(self.current_alert_group)
        
        for ip, severities in alerts_by_ip.items():
            self._update_ip_metrics(ip, severities)

        if self.current_node == self.end_node: 
            terminal_reward = 100
            return next_node, terminal_reward, alert_group_t, True
        
        return next_node, reward, alert_group_t, False
    
    def get_actions(self, node):
        return list(self.graph.successors(node))
    
    def initialize_from_json(self):
        #data = json.load(self.json_graph)
        data = self.json_graph
        G = nx.DiGraph()

        name_to_id = {}

        for asset_id, asset_data in data['assets'].items():
            name = asset_data['name']
            G.add_node(asset_id, id=asset_id, name=name, metaconcept=asset_data['metaconcept'])
            name_to_id[name] = asset_id  

        for asset_id, asset_data in data['assets'].items():

            style_attrs = {}

            detection_prob = 0.0
            base_severity = 4
            alert_type = 'unknown'

            name = asset_data['name']

            if asset_data['metaconcept'] == 'Vulnerability':
                style_attrs = {'style': 'filled', 'fillcolor': 'red'}
                detection_prob = 0.8 
                base_severity = 4
                alert_type = 'network'
            elif asset_data['metaconcept'] == 'Privileges':
                style_attrs = {'style': 'filled', 'fillcolor': 'blue'}
                detection_prob = 0.4 
                base_severity = 5
                alert_type = 'local'
            elif name.startswith("Reconnaissance"):
                detection_prob = 0.6
                base_severity = 3
                alert_type = 'recon'
            elif name.startswith("Network_"):
                detection_prob = 0.7
                base_severity = 2
                alert_type = 'network'
            elif name.startswith("HostCompromise"):
                detection_prob = 0.85
                base_severity = 3
                alert_type = 'network'

            G.add_node(asset_id, 
                       id=asset_id, 
                       name=asset_data['name'], 
                       metaconcept=asset_data['metaconcept'], 
                       detection_prob=detection_prob,
                       base_severity=base_severity, 
                       alert_type=alert_type, 
                       **style_attrs)

        for association in data['associations']:
                    assoc_data = association['association']
                    end_values = [v[0] for v in assoc_data.values()]
                    
                    value = self.reward_model.NON_ACTION_VALUE

                    #print(f"DEBUG: The default step penalty is: {value}")
                    # Find the CVE and its position (0 or 1) in the association
                    for i, node_id in enumerate(end_values):
                        if G.nodes[str(node_id)]['name'].startswith("CVE"):
                            node_name = G.nodes[str(node_id)]['name']
                            cve = self.extract_cve_from_node_name(node_name)
                            cvss_string = self.get_cvss_by_cve(self.DATABASE_PATH, cve)
                            
                            if cvss_string: # and cvss_string != 'Empty':
                                self.total_vuln += 1
                                # Ask the model for the reward, providing crucial context.
                                value = self.reward_model.get_reward(
                                    cvss_string=cvss_string,
                                    node_position=i
                                )
                                print(f"DEBUG: {cvss_string} The default step penalty is: {value} | SOURCE is {end_values[0]} DESTINATION is {end_values[1]}")
                            break 
                    #print(f"DEBUG: The default step penalty is: {value} | SOURCE is {end_values[0]} DESTINATION is {end_values[1]}")
                    G.add_edge(end_values[0], end_values[1], distance = value)

                    #G.add_edge(end_values[1], end_values[0], distance=-2 )

        return G, nx.adjacency_matrix(G),name_to_id

    def compute_node_metrics(self):
        self.centrality = nx.betweenness_centrality(self.graph)

    def graphvizualise(self):
        self.graph['graph']={'rankdir':'TD'}
        self.graph['node']={'shape':'circle'}
        self.graph['edges']={'arrowsize':'4.0'}
        AG = to_agraph(self.graph)
        AG.draw('graph-visualisation/graph.png', prog='dot')
        return

    def visualise_interactive_streamlit(self, current_node=None, previous_node=None, history=None):

        if not self.graph:
            return "<h3>Graph is not initialized.</h3>"

        if history is None:
            history = []

        path_edges = set(zip(history, history[1:]))
        centrality = nx.degree_centrality(self.graph)

        net = Network(height="1200px", width="100%", notebook=False, directed=True, bgcolor="#222222", font_color="white")

        options = """
        var options = {
        "physics": {
            "forceAtlas2Based": {
            "gravitationalConstant": -25,
            "centralGravity": 0.01,
            "springLength": 230,
            "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": { "iterations": 150 }
        },
        "nodes": {
            "font": {
            "size": 16,
            "color": "white"
            }
        }
        }
        """
        net.set_options(options)

        for node, data in self.graph.nodes(data=True):
            node_id = str(node)
            node_name = data.get('name', node_id)
            
            node_size = 15 + (centrality.get(node, 0) * 50)
            node_title = f"{node_name}<br> | node ID: {node_id}"
            color = '#97c2fc'

            if node_id == str(current_node):
                color = '#2ecc71'
                node_size = max(node_size, 40)
            elif node_id == str(previous_node):
                color = '#f1c40f'
                node_size = max(node_size, 30)
                

            net.add_node(node_id, label=node_name, title=node_title, size=node_size, color=color)

        for u_node, v_node in self.graph.edges():
            u_id, v_id = str(u_node), str(v_node)
            if (u_id, v_id) in path_edges:
                net.add_edge(u_id, v_id, width=4, color='#f5b041')
            else:
                net.add_edge(u_id, v_id, width=1, color='#444444')
                
        return net.generate_html(notebook=False)


    # def print_nodes(self):
    #     for node in self.graph['node']:
    #         print(node)
    #     return

    def pretty_print(self):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', width=2)
        edge_labels = nx.get_edge_attributes(self.graph, 'distance')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        node_labels = {node: node.split(',')[0] for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels)
        plt.axis('off')
        plt.show()

    def parse_cvss(self, cvss_string):
        cvss_metrics = {
            'AV': 'Attack Vector',
            'AC': 'Attack Complexity',
            'PR': 'Privileges Required',
            'UI': 'User Interaction',
            'S': 'Scope',
            'C': 'Confidentiality',
            'I': 'Integrity',
            'A': 'Availability',
            'E': 'Exploit Code Maturity',
            'RL': 'Remediation Level',
            'RC': 'Report Confidence',
            'CR': 'Confidentiality Requirement',
            'IR': 'Integrity Requirement',
            'AR': 'Availability Requirement',
            'MAV': 'Modified Attack Vector',
            'MAC': 'Modified Attack Complexity',
            'MPR': 'Modified Privileges Required',
            'MUI': 'Modified User Interaction',
            'MS': 'Modified Scope',
            'MC': 'Modified Confidentiality',
            'MI': 'Modified Integrity',
            'MA': 'Modified Availability'
        }
        
        vector = {
            'N': 0.85,
            'A': 0.62,
            'L': 0.55,
            'P': 0.20,
        }
        complexity = {
            'L': 0.77,
            'H': 0.44,
        }
        privileges = {
            'N': 0.85,
            'L': 0.62,
            'H': 0.27
        }
        interaction = {
            'N': 0.85,
            'R': 0.62,
        }
        cia = {
            'N': 0.00,
            'L': 0.22,
            'H': 0.56
        }
        if cvss_string.startswith("CVSS:3.1/"):
            cvss_string = cvss_string[len("CVSS:3.1/"):]

        metric_pairs = cvss_string.split('/')
        
        parsed_qn_metrics = {}
        for pair in metric_pairs:
            #print("-------11111------", pair)
            metric, value = pair.split(':')
            if metric == 'AV':
                parsed_qn_metrics['AV']=vector.get(value, 'Unknown Metric')
            elif metric == "AC":
                parsed_qn_metrics['AC']=complexity.get(value, 'Unknown Metric')
            elif metric == "PR":
                parsed_qn_metrics['PR']=privileges.get(value, 'Unknown Metric')
            elif metric == "UI":
                parsed_qn_metrics['UI'] = interaction.get(value, 'Unknown Metric')
            elif metric == "C":
                parsed_qn_metrics['C'] = cia.get(value, 'Unknown Metric')
            elif metric == "A":
                parsed_qn_metrics['A'] = cia.get(value, 'Unknown Metric')
            elif metric == "I" :
                parsed_qn_metrics['I'] = cia.get(value, 'Unknown Metric')           
            else:
                pass

        return parsed_qn_metrics

    def get_cvss_string(self, cve_id):
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
        # url = f"https://services.nvd.nist.gov/rest/json/cve/2.0/"
        #print(url)

        headers = {
            'apiKey': self.NVD_KEY
        }
    
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()

                if 'result' in data and 'CVE_Items' in data['result']:
                    cve_item = data['result']['CVE_Items'][0]

                    if 'impact' in cve_item and 'baseMetricV3' in cve_item['impact']:
                        cvss_v3 = cve_item['impact']['baseMetricV3']['cvssV3']
                        cvss_string = f"CVSS v3.1: {cvss_v3['baseScore']} ({cvss_v3['baseSeverity']})"
                        return cvss_string

                    elif 'impact' in cve_item and 'baseMetricV2' in cve_item['impact']:
                        cvss_v2 = cve_item['impact']['baseMetricV2']['cvssV2']
                        cvss_string = f"CVSS v2.0: {cvss_v2['baseScore']} ({cvss_v2['severity']})"
                        return cvss_string
                    else:
                        return "CVSS score not available"
                else:
                    return "Invalid CVE ID or no data found"
            else:
                return f"Failed to retrieve data, HTTP status code: {response.status_code}"

        except Exception as e:
            return f"An error occurred: {e}"

    def get_cvss_by_cve(self, database_path, cve):
        try:
            conn = sqlite3.connect(database_path)
            #print(database_path)
            cursor = conn.cursor()

            query = '''
            SELECT cvss_string FROM vulnerability WHERE cve = ?
            '''

            cursor.execute(query, (cve,))

            result = cursor.fetchone()

            if result:
                cvss_id = result[0]
                #print(f"CVSS ID for CVE {cve}: {cvss_id}")
                conn.close()
                return cvss_id
            else:
                #print(f"No CVSS ID found for CVE {cve}")
                conn.close()
                return 'Empty'
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

    def find_node_with_name_from_predecessors(self, start_node, target_name):
        visited = set() 
        stack = [start_node] 

        while stack:
            current_node = stack.pop()

            if current_node in visited:
                continue

            visited.add(current_node)

            if self.graph.nodes[str(start_node)]['name'].startswith("Reconnaisance"):
                current_node = self.graph.nodes[str(start_node)]
                match = re.search(r"Host \d+", self.graph.nodes[str(self.current_node)]['name'])
                host = match.group() if match else None
                print(f"Found target node '{target_name}' at node {current_node}")
                return host, current_node

            predecessors = list(self.graph.predecessors(current_node))
            stack.extend(predecessors)

        #print(f"No node with name '{target_name}' found.")
        return None, None

    def extract_cve_from_node_name(self, node_name):
        match = re.search(r'(CVE-\d{4}-\d{4,7})', node_name, re.IGNORECASE)
        return match.group(1) if match else None

    def find_node_ids_by_names(self, target_names: list):
        return [self.name_to_id_map.get(name) for name in target_names]

    def get_adjacency_bitmap(self, binary=False):
        current_matrix = self._compute_masked_adjacency()
        if binary:
            return (current_matrix > 0).astype(int)
        return np.array(current_matrix)
    
    def reset_mask(self):
        self.node_mask.fill(1)
        self.edge_mask.fill(1)

        self.removed_edges = [] 
        self.adjacency_matrix = self._compute_masked_adjacency()
        
        return self.set_current_node(self.nodes[0])

    def _compute_masked_adjacency(self):
        masked = np.multiply(self.base_adjacency, self.edge_mask)
        node_mask_2d = np.outer(self.node_mask, self.node_mask)
        final_matrix = np.multiply(masked, node_mask_2d)
        return final_matrix
    
    def compute_and_update_risk(self, decay_factor=0.1, max_iterations=100, tolerance=1e-6):
        """
        computes the unconditional probability of compromise with Forward Propagation 
        """
        G = self.graph
        nodes = list(G.nodes())
        
        node_probs = {}
        
        for node in nodes:
            if G.in_degree(node) == 0:
                node_probs[node] = 1.0
            else:
                node_probs[node] = 0.0

        edge_probs = {}
        for u, v, data in G.edges(data=True):
            u_idx, v_idx = self.node_to_idx[u], self.node_to_idx[v]
                
            if self.edge_mask[u_idx, v_idx] == 0:
                p_edge = 0.0
            else:
                dist = data.get('distance', self.reward_model.NON_ACTION_VALUE)
                p_edge = np.exp(-decay_factor * dist)
            
            edge_probs[(u, v)] = p_edge

        for i in range(max_iterations):
            max_change = 0.0
            new_probs = node_probs.copy()
            
            targets = [n for n in nodes if G.in_degree(n) > 0]
            
            for u in targets:
                predecessors = list(G.predecessors(u))
                
                if not predecessors:
                    continue

                prob_complement_product = 1.0
                
                for p in predecessors:
                    p_parent = node_probs[p]              
                    p_transition = edge_probs.get((p, u), 0) 
                    
                    p_attack_success = p_parent * p_transition
                    
                    prob_complement_product *= (1.0 - p_attack_success)
                
                calculated_prob = 1.0 - prob_complement_product
                max_change = max(max_change, abs(calculated_prob - node_probs[u]))
                new_probs[u] = calculated_prob
            
            node_probs = new_probs
            
            if max_change < tolerance:
                break

        for node, prob in node_probs.items():
            self.graph.nodes[node]['unconditional_risk'] = prob

        # print(f"Risk analysis (Forward Propagation) complete. Converged in {i+1} iterations.")
        return node_probs
    
    def list_risk_values(self, top_n=None):
        """
        Lists the unconditional risk probabilities for nodes, sorted by highest risk.
        Must run compute_and_update_risk() first.
        """
        risk_data = []
        for node, data in self.graph.nodes(data=True):
            prob = data.get('unconditional_risk', 0.0)
            name = data.get('name', 'Unknown')
            risk_data.append({'id': node, 'name': name, 'risk': prob})

        sorted_data = sorted(risk_data, key=lambda x: x['risk'], reverse=True)

        if top_n:
            sorted_data = sorted_data[:top_n]

        print(f"\n{'='*60}")
        print(f"{'PROBABILISTIC RISK REPORT':^60}")
        print(f"{'='*60}")
        print(f"{'PROBABILITY':<12} | {'NODE ID':<10} | {'NODE NAME'}")
        print(f"{'-'*12} | {'-'*10} | {'-'*30}")

        for item in sorted_data:
            score_display = f"{item['risk']:.4f}"
            
            if item['risk'] == 1.0 and self.graph.in_degree(item['id']) == 0:
                score_display += " (ENTRY)"
            
            print(f"{score_display:<12} | {str(item['id']):<10} | {item['name']}")
        
        print(f"{'='*60}\n")

    def apply_action(self, action_id, target_node=None, **kwargs):
        """
        Executes the defensive action on the SPECIFIC target node chosen by the GNN.
        """
        if action_id == 0:
            return self._action_do_nothing()
            
        elif action_id == 1:
            return self._action_network_filtering(source_node=target_node, **kwargs)
            
        elif action_id == 2:
            return self._action_restore_software(target_node_id=target_node, **kwargs)
            
        elif action_id == 3:
            return self._action_restore_network_connection(target_node_id=target_node, **kwargs)
            
        elif action_id == 4:
            if target_node:
                node_name = self.graph.nodes[target_node].get('name', '')
                return self._action_isolate_ip(ip_address=node_name)
            return {"status": "failed", "message": "No target node provided for isolation"}
            
        else:
            return {"status": "error", "message": f"Invalid action ID: {action_id}"}

    def _action_do_nothing(self, debug=False):
        if debug: print("Action applied: Do Nothing.")
        return {"status": "success", "action": "Do Nothing", "changes": 0}
    
    def _action_network_filtering(self, source_node, debug=False):
        if source_node is None:
            if debug: print(f"  -> [Invalid] Action Failed: No target node specified.")
            return {"status": "failed", "message": "No target node specified"}

        u_idx = self.node_to_idx.get(source_node)
        successors = list(self.graph.successors(source_node))
        active_edges = 0
        
        if u_idx is not None:
            for v_node in successors:
                v_idx = self.node_to_idx.get(v_node)
                if self.edge_mask[u_idx, v_idx] == 1:
                    active_edges += 1

        if active_edges == 0:
            if debug: print(f"  -> [Invalid] Context: Filtering {source_node} has no effect (No active outgoing edges).")
            return {"status": "failed", "action": "Network Filtering", "changes": 0, "valid_context": False}

        changes_count = 0
        status = "failed"
        
        for v_node in successors:
            v_idx = self.node_to_idx.get(v_node)
            if self.edge_mask[u_idx, v_idx] == 1:
                self.edge_mask[u_idx, v_idx] = 0
                self.removed_edges.append((source_node, v_node))
                changes_count += 1
        
        if changes_count > 0:
            status = "success"
            self.adjacency_matrix = self._compute_masked_adjacency()
            if debug:
                print(f"Action applied: Network Filtering on {source_node}. Blocked {changes_count} outgoing edges.")

        self.defensive_action_history.append({
            "step": len(self.history),
            "action": "Network Filtering",
            "target": str(source_node),
            "status": status,
            "changes": changes_count,
            "valid_context": True 
        })

        return {
            "status": status, 
            "action": "Network Filtering", 
            "changes": changes_count,
            "valid_context": True
        }

    def _action_restore_software(self, target_node_id=None, ip_address=None, cve_id=None, debug=False):
        nodes_to_mask = []

        if target_node_id:
            if target_node_id not in self.graph.nodes:
                return {"status": "failed", "message": "Node ID not found"}

            node_name = self.graph.nodes[target_node_id].get('name', '')
            
            if "CVE" in node_name:
                nodes_to_mask.append(target_node_id)
            else:
                successors = list(self.graph.successors(target_node_id))
                predecessors = list(self.graph.predecessors(target_node_id))
                neighbors = successors + predecessors 
                
                for neighbor in neighbors:
                    n_name = self.graph.nodes[neighbor].get('name', '')
                    if "CVE" in n_name:
                        nodes_to_mask.append(neighbor)
        
        elif ip_address:
             for node, data in self.graph.nodes(data=True):
                name = data.get('name', '')
                if ip_address in name and "CVE" in name:
                    nodes_to_mask.append(node)

        if not nodes_to_mask:
            if debug: print(f"  -> [Invalid] Context: Target {target_node_id} has no attached CVEs to patch.")
            return {
                "status": "failed", 
                "action": "Restore Software", 
                "changes": 0, 
                "valid_context": False
            }

        removed_count = 0
        for node in nodes_to_mask:
            n_idx = self.node_to_idx.get(node)
            
            if n_idx is not None and self.node_mask[n_idx] == 1:
                self.node_mask[n_idx] = 0 
                removed_count += 1
                if debug:
                    print(f"  -> Patched/Masked node {node} ({self.graph.nodes[node].get('name')})")

        if removed_count > 0:
            self.adjacency_matrix = self._compute_masked_adjacency()

        status = "success" if removed_count > 0 else "failed"
        self.defensive_action_history.append({
            "step": len(self.history),
            "action": "Restore Software",
            "target": str(target_node_id),
            "status": status,
            "changes": removed_count,
            "valid_context": True
        })

        return {
            "status": status, 
            "action": "Restore Software", 
            "changes": removed_count, 
            "valid_context": True
        }
    
    def _action_restore_network_connection(self, target_node_id=None, debug=False):
        if not self.removed_edges:
            return {"status": "success", "message": "No edges to restore."}

        edge_to_restore = None
        restore_index = -1

        if target_node_id:
            for i, (u, v) in enumerate(self.removed_edges):
                if u == target_node_id or v == target_node_id:
                    edge_to_restore = (u, v)
                    restore_index = i
                    break
        
        if edge_to_restore is None:
             if debug: print("  -> No specific edge found for target. Fallback to LIFO.")
             edge_to_restore = self.removed_edges[-1]
             restore_index = len(self.removed_edges) - 1

        u, v = edge_to_restore
        
        del self.removed_edges[restore_index]
        
        u_idx = self.node_to_idx.get(u)
        v_idx = self.node_to_idx.get(v)
        
        if u_idx is not None and v_idx is not None:
            self.edge_mask[u_idx, v_idx] = 1 
            self.adjacency_matrix = self._compute_masked_adjacency()
            
            if debug: print(f"  -> Restored connection {u} -> {v}.")

            return {"status": "success", "restored": (u, v), "changes": 1}
            
        return {"status": "error", "message": "Node indices not found."}

    def _action_isolate_ip(self, ip_address, debug=False):
        if not ip_address or not isinstance(ip_address, str):
            match = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", str(ip_address))
            if match:
                ip_address = match.group()
            else:
                return {"status": "failed", "message": "Invalid IP address string"}

        nodes_to_isolate = []
        for node, data in self.graph.nodes(data=True):
            if ip_address in data.get('name', ''):
                nodes_to_isolate.append(node)
        
        blocked_count = 0
        
        for u_node in nodes_to_isolate:
            u_idx = self.node_to_idx.get(u_node)
            if u_idx is None: continue

            for v_node in self.graph.successors(u_node):
                v_idx = self.node_to_idx.get(v_node)
                if self.edge_mask[u_idx, v_idx] == 1:
                    self.edge_mask[u_idx, v_idx] = 0
                    self.removed_edges.append((u_node, v_node))
                    blocked_count += 1
            
            for v_node in self.graph.predecessors(u_node):
                v_idx = self.node_to_idx.get(v_node)
                if self.edge_mask[v_idx, u_idx] == 1:
                    self.edge_mask[v_idx, u_idx] = 0
                    self.removed_edges.append((v_node, u_node))
                    blocked_count += 1

        if blocked_count > 0:
            self.adjacency_matrix = self._compute_masked_adjacency()
            if debug:
                print(f"Action applied: Isolate IP {ip_address}...")

        status = "success" if blocked_count > 0 else "failed"
        
        self.defensive_action_history.append({
            "step": len(self.history),
            "action": "Isolate IP",
            "target": str(ip_address),
            "status": status,
            "changes": blocked_count,
            "valid_context": True
        })

        return {
            "status": status, 
            "action": "Isolate IP",
            "nodes_isolated": len(nodes_to_isolate), 
            "blocked_edges": blocked_count,
            "changes": blocked_count 
        }
    
    def get_action_history(self):
        """Returns the log of defensive actions taken."""
        return self.defensive_action_history
    

    # PLEASE GOD WORK

    def get_graph_observation(self):

        import torch
        
        node_features = []
        
        if not self.centrality:
            self.compute_node_metrics()
            
        for node in self.nodes:
            idx = self.node_to_idx[node]
            f_active = float(self.node_mask[idx])
            
            f_risk = float(self.graph.nodes[node].get('unconditional_risk', 0.0))
            f_central = float(self.centrality.get(node, 0.0))
            
            node_name = self.graph.nodes[node].get('name', '')
            f_vuln = 1.0 if "CVE" in node_name else 0.0
            
            f_crit = 1.0 if str(node) == str(self.end_node) else 0.0
            
            raw_z = self.graph.nodes[node].get('z_score', 0.0)
            f_zscore = max(-3.0, min(3.0, raw_z)) / 3.0
            
            raw_ent = self.graph.nodes[node].get('entropy', 0.0)
            f_entropy = np.tanh(raw_ent)

            f_src_vlm = self.graph.nodes[node].get('alert_volume_src', 0.0)

            f_dst_vlm = self.graph.nodes[node].get('alert_volume_dst', 0.0)
            
            f_is_alerted = 1.0 if self.graph.nodes[node].get('status') == 'Alerted' else 0.0

            node_features.append([f_active, f_risk, f_central, f_vuln, f_crit, f_zscore, f_entropy, f_is_alerted, f_src_vlm, f_dst_vlm])
            
            # if self.graph.nodes[node].get('entropy', 0.0) != 0:
            #     print(self.graph.nodes[node].get('name'))
            #     print("entropy = ", self.graph.nodes[node].get('entropy', 0.0))
            #     print("src-vlm = ", self.graph.nodes[node].get('alert_volume_src', 0.0))
            #     print("dst-vlm = ", self.graph.nodes[node].get('alert_volume_dst', 0.0))
            # if self.graph.nodes[node].get('z_score', 0.0) != 0:
            #     #print(self.graph.nodes[node].get('name'))
            #     print("ZScore = ", self.graph.nodes[node].get('z_score', 0.0))
                      
        x = torch.tensor(node_features, dtype=torch.float)
        
        edges = []
        for u, v in self.graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                edges.append([self.node_to_idx[u], self.node_to_idx[v]])
                
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return x, edge_index

    def get_valid_action_mask(self):
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        alerted_ips = set()

        ip_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
        # print("LEN = ", len(self.current_alert_group))
        #print(self.current_alert_group)
        for alert_str in self.current_alert_group:
            
            found_ips = ip_pattern.findall(str(alert_str))
            #print(found_ips)

            for ip in found_ips:
                #print(found_ips)
                if ip in self.ip_to_nodes_map:
                    if self.ip_to_nodes_map[ip] != []:
                        temp_node = self.graph.nodes[self.ip_to_nodes_map[ip][0]]
                        if temp_node.get("alert_volume_src") > 0.05 or temp_node.get("alert_volume_dst") > 0.05:
                            # print("--------->",ip)
                            alerted_ips.add(ip)

        alert_found = False

        # print("IPS ---- ALERTED ------------------")
        # print(alerted_ips)
        # print("-----------------------------------")

        for idx, node_id in enumerate(self.nodes):
            node_name = self.graph.nodes[node_id].get('name', '')
            for ip in alerted_ips:
                if ip in node_name:
                    # print("ip is masked - : ", ip )
                    # print("----------------")
                    mask[idx] = True
                    alert_found = True
                    break 
                
        if not alert_found:
            mask[0] = True
                
        return mask
