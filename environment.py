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
import sqlite3
import numpy as np
from collections import deque, defaultdict
from utils.Alert import AlertGenerator
from utils.Colors import BColors
import re
import logging
from rewards.RewardModel import RewardModel, DefaultRewardModel


def load_config(path='config.json'):
    """Loads the JSON configuration file."""
    with open(path, 'r') as f:
        return json.load(f)
  

class GraphEnvironment:

    def __init__(self, json):
        #INITILIAZE
        logging.basicConfig(filename="logs/env_logs.log",
                            encoding="utf-8",
                            filemode="a+",
                            format = "{asctime} - {levelname} - {message}",
                            style = "{",
                            datefmt = "%Y-%m-%d %H:%M",
                            level=logging.INFO)

        self.reward_model = DefaultRewardModel()
        config = load_config()
        env_config = config['environment_settings']
        self.NORM_F = env_config['NORM_F']['value']
        self.NORM_F_ATTEMPT = env_config['NORM_F_ATTEMPT']['value']
        self.NON_ACTION_VALUE = env_config['NON_ACTION_VALUE']['value']
        self.NVD_KEY = env_config['NVD_KEY']['value']
        self.DATABASE_PATH = env_config['DATABASE_PATH']['value']

        self.removed_edges = []
        
        self.json_graph = json
        self.alert_gen = AlertGenerator()
        self.vulnerability_scores = {}
        self.alert_history = deque(maxlen=50)
        self.centrality = {}
        self.history = []
        self.vuln_met = 0
        self.total_vuln = 0

        self.graph, self.adjacency_matrix, self.name_to_id_map = self.initialize_from_json() #NetworkX graph
        self._update_graph_state()


        #NODES and NUM_NODES
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        self.end_node = self.nodes[-1]
        self.end_node = '307'

        #self.end_node = self.nodes[-3]
        self.current_alert = ""
        self.noise_prob = 0.05
        self.realistic_alerts = 1

        self.volume = 30
        self.offset_a = 20

        self.current_alert_group = []

        #This probabably needs to change with the graph model that comes as an inout
        self.current_node = 0
        self.previous_node = 0
        self.current_host = 0
        self.previous_host = -1

    def _update_graph_state(self):
        """
        We updates all graph-derived attributes after a modification especiually with the new dedicated Action functions.
        """
        print("...Updating graph state (nodes, adjacency matrix)...")
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        self.adjacency_matrix = nx.adjacency_matrix(self.graph)

    def reset(self): #REQUIRED FOR EVERY NEW INSTANCE
        #self.current_node = random.choice(self.nodes)
        self.current_host = 0
        self.previous_host = -1

        self.previous_node = 0
        self.current_node = self.nodes[0]
        self.current_alert_group = []

        self.vuln_met = 0

        return self.current_node

    def random_reset(self): # RANDOM RESET IS BROKEN FIX BEFORE START CODE
        #self.current_node = random.choice(self.nodes)
        self.current_host = 0
        self.previous_host = -1

        self.previous_node = 0

        self.current_alert_group = []

        self.current_node = random.choice(list(self.graph.nodes()))
        
        self.current_host, node_of_c = self.find_node_with_name_from_predecessors(self.current_node, "Reconnaisance")

        predecessors = list(self.graph.predecessors(self.current_node))

        if predecessors:
            self.previous_node = random.choice(predecessors)
        else:
            self.previous_node = 0

        self.vuln_met = 0

        return self.current_node

    def set_current_node(self, node):
        self.current_node = node
        return self.current_node
    
    # MODEL_BASED ADJUSTMENTS NEED TO BE DONE HERE
    def step(self, action):

        next_node = action

        self.current_alert = ""
        new_state_information = []

        if next_node not in self.graph[self.current_node]:
            # If action (next_node) is not a valid successor, end the episode
            return self.current_node, -1000, self.current_alert, True  # High negative reward for invalid action
        
        
        if self.graph.nodes[str(self.current_node)]['name'].startswith("Reconnaisance"):

            match = re.search(ip_pattern, self.graph.nodes[str(self.current_node)]['name'])
            self.current_host = match.group() if match else None

        

        distance = self.graph[self.current_node][next_node]['distance']
        reward = distance

        self.current_alert_group = []

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        if self.reward_model.is_vulnerability_met(self.graph, self.current_node):
            self.vuln_met += 1
        # -------------------------------------------------------------


        self.previous_node = self.current_node
        self.current_node = next_node
        self.history.append(self.current_node)

        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"

        #At this stage I get the data from my changed network X node
        current_node_data = self.graph.nodes[str(self.current_node)]
        current_node_name = current_node_data.get('name', '')
        
        # Parsin gthe newly added info and the alert type
        base_prob = current_node_data.get('detection_prob', 0.0)
        base_sev = current_node_data.get('base_severity', 4)

        alert_type = current_node_data.get('alert_type', 'unknown')

        if self.realistic_alerts == 1:        
            range_num = severity = random.randint(self.volume-self.offset_a, self.volume+self.offset_a)

        # at this point the hopefull PhD student will try to generate alerts (WE DID IT FFS : or a volume of alerts but this is something to look on a later stage)
        for item in range (1, range_num):

            if random.random() < base_prob:

                # Get DESTINATION IP (from the new/current node)
                match = re.search(ip_pattern, current_node_name)
                dest_ip_for_alert = match.group() if match else None
                
                # Get SOURCE IP (from the previous node)
                prev_node_name = self.graph.nodes[str(self.previous_node)]['name']
                match = re.search(ip_pattern, prev_node_name)
                source_ip_for_alert = match.group() if match else None


                # hopeflly we do not reach that point but this a failsae mecanisms. If i generate this shit, then i have an important problem
                if self.previous_node == 0:
                    source_ip_for_alert = "13.12.4.20" 

                if not dest_ip_for_alert:
                    dest_ip_for_alert = source_ip_for_alert
                
                if not source_ip_for_alert:
                    source_ip_for_alert = "13.12.4.20" # Failsafe
    
                self.previous_host = source_ip_for_alert
                self.current_host = dest_ip_for_alert

                if alert_type == 'recon':
                    self.current_alert = self.alert_gen.generate_alert_recon(
                        source_ip_for_alert, random.randint(1024, 65535), 
                        dest_ip_for_alert, 80, # e.g., scanning port 80
                        severity=base_sev
                    )
                
                elif alert_type == 'local':
                    self.current_alert = self.alert_gen.generate_alert_local(
                        dest_ip_for_alert, random.randint(1024, 65535), 
                        dest_ip_for_alert, random.randint(1024, 65535), # Src and Dest are same
                        severity=base_sev
                    )

                elif alert_type == 'network':
                    # for CVEs, we use the CVSS-based 'distance' for severity
                    if "CVE" in current_node_name:
                        edge_data = self.graph[self.previous_node][self.current_node]
                        cvss_based_value = edge_data.get('distance', -1)
                        if cvss_based_value > 8: base_sev = 5 # Critical
                        elif cvss_based_value > 6: base_sev = 4 # Medium
                    
                    self.current_alert = self.alert_gen.generate_alert_network(
                        source_ip_for_alert, random.randint(1024, 65535), 
                        dest_ip_for_alert, 445, # e.g., exploiting SMB
                        severity=base_sev
                    )

                self.current_alert_group.append(self.current_alert)
                self.current_alert = ''

            if not self.current_alert and random.random() < self.noise_prob:
                self.current_alert = self.alert_gen.generate_alert_noise()

        alert_group_t = self.current_alert_group
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        if self.current_node == self.end_node: 
            terminal_reward = self.reward_model.get_reward(
                is_terminal=True,
                vuln_met=self.vuln_met,
                total_vuln=self.total_vuln
            )
            #terminal_reward = 100
        # -------------------------------------------------------------
            return next_node, terminal_reward, alert_group_t, True
        


        # Check info to generate alert

        # if self.graph.nodes[str(self.current_node)]['name'].startswith("Reconnaissance"):
        #     # Assign the previous to be the one that was previous discovered in the fist condition
        #     self.previous_host = self.current_host
        #     # Match String to find the new host since the current new host has now changed
        #     match = re.search(ip_pattern, self.graph.nodes[str(self.current_node)]['name'])
        #     self.current_host = match.group() if match else None
        #     # TODO: CHANGE ACCORDINGLY WHEN THE SL300 IS IMPLEMENTED
        #     dest_ip, src_port, src_ip, dest_port = self.current_host, 8001, self.previous_host, 443
        #     self.current_alert = self.alert_gen.generate_alert_recon(src_ip, src_port, dest_ip, dest_port)

        # if self.graph.nodes[str(self.current_node)]['name'].startswith("Local_"):
        #     # TODO: CHANGE ACCORDINGLY WHEN THE SL300 IS IMPLEMENTED
        #     dest_ip, src_port, src_ip, dest_port = self.current_host, 8001, self.previous_host, 443
        #     self.current_alert = self.alert_gen.generate_alert_local(src_ip, src_port, dest_ip, dest_port)

        # if self.graph.nodes[str(self.current_node)]['name'].startswith("Network_"):
        #     # TODO: CHANGE ACCORDINGLY WHEN THE SL300 IS IMPLEMENTED
        #     dest_ip, src_port, src_ip, dest_port = self.current_host, 8001, self.previous_host, 443
        #     self.current_alert = self.alert_gen.generate_alert_network(src_ip, src_port, dest_ip, dest_port)

        # logging.info("\n")
        # logging.info(self.graph.nodes[str(self.current_node)]['name'])
        # logging.info("-------------------------------------------------------")
        # logging.info(f"Current Host: {self.current_host} - The host currently in use.")
        # logging.info(f"Previous Host: {self.previous_host} - The host used prior to the current one.")
        # logging.info(f"Previous Node: {self.previous_node} - The node accessed before the current node.")
        # logging.info(f"Current Node: {self.current_node} - The node currently being processed.")
        # logging.info(BColors.WARNING + json.dumps(self.current_alert, indent=4) + BColors.ENDC)
        
        # print("\n")
        # print(self.graph.nodes[str(self.current_node)]['name'])
        # print("-------------------------------------------------------")
        # print(f"Current Host: {self.current_host} - The host currently in use.")
        # print(f"Previous Host: {self.previous_host} - The host used prior to the current one.")
        # print(f"Previous Node: {self.previous_node} - The node accessed before the current node.")
        # print(f"Current Node: {self.current_node} - The node currently being processed.")
        # print(BColors.WARNING + json.dumps(self.current_alert, indent=4) + BColors.ENDC)


        return next_node, reward, alert_group_t, False  # For simplicity, assume no terminal state yet
    
    def get_actions(self, node):
        return list(self.graph.successors(node))
    
    def initialize_from_json(self):

        #data = json.load(self.json_graph)
        data = self.json_graph
        G = nx.DiGraph()


        name_to_id = {}

        # Add nodes and populate the map
        for asset_id, asset_data in data['assets'].items():
            name = asset_data['name']
            G.add_node(asset_id, id=asset_id, name=name, metaconcept=asset_data['metaconcept'])
            name_to_id[name] = asset_id  # Map the name to its ID

        # Add nodes
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
            # --- END NEW ---

            G.add_node(asset_id, 
                       id=asset_id, 
                       name=asset_data['name'], 
                       metaconcept=asset_data['metaconcept'], 
                       detection_prob=detection_prob,  # <-- ADDED
                       base_severity=base_severity,    # <-- ADDED
                       alert_type=alert_type,          # <-- ADDED
                       **style_attrs)


        # Add edges
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
                            break # Stop after finding the first CVE in the association
                    #print(f"DEBUG: The default step penalty is: {value} | SOURCE is {end_values[0]} DESTINATION is {end_values[1]}")
                    G.add_edge(end_values[0], end_values[1], distance=value)

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

        # --- OPTIONS FOR PERFORMANCE AND FONT ---
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
        # -----------------------------------------------

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
                
            # *** MODIFIED LINE ***
            # Add the 'label=node_name' parameter to make the name always visible.
            # The 'title' property is kept for the detailed hover tooltip.
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
        # Remove the CVSS:3.1/ prefix if present
        if cvss_string.startswith("CVSS:3.1/"):
            cvss_string = cvss_string[len("CVSS:3.1/"):]

        # Split the string into individual metric-value pairs
        metric_pairs = cvss_string.split('/')
        
        # Parse the metric-value pairs
        parsed_qn_metrics = {}
        for pair in metric_pairs:
            #print("-------11111------", pair)
            metric, value = pair.split(':')
            # Translate the metric abbreviations to their full names using the cvss_metrics dictionary
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
        # NVD API URL for the specific CVE
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
        # url = f"https://services.nvd.nist.gov/rest/json/cve/2.0/"
        #print(url)

        headers = {
            'apiKey': self.NVD_KEY
        }
    
        try:
            # Send a GET request to the NVD API
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()

                # Extract the CVSS v3 string if available
                if 'result' in data and 'CVE_Items' in data['result']:
                    cve_item = data['result']['CVE_Items'][0]

                    # Check if CVSS v3 is available
                    if 'impact' in cve_item and 'baseMetricV3' in cve_item['impact']:
                        cvss_v3 = cve_item['impact']['baseMetricV3']['cvssV3']
                        cvss_string = f"CVSS v3.1: {cvss_v3['baseScore']} ({cvss_v3['baseSeverity']})"
                        return cvss_string

                    # Fallback to CVSS v2 if v3 is not available
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
            # Connect to the SQLite database
            conn = sqlite3.connect(database_path)
            #print(database_path)
            cursor = conn.cursor()

            # Query to get the cvss based on the provided cve
            query = '''
            SELECT cvss_string FROM vulnerability WHERE cve = ?
            '''

            # Execute the query with the provided cve
            cursor.execute(query, (cve,))

            # Fetch the result
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
        visited = set()  # To avoid revisiting nodes
        stack = [start_node]  # Stack for DFS

        while stack:
            current_node = stack.pop()

            # Skip already visited nodes
            if current_node in visited:
                continue

            visited.add(current_node)

            if self.graph.nodes[str(start_node)]['name'].startswith("Reconnaisance"):
                current_node = self.graph.nodes[str(start_node)]
                match = re.search(r"Host \d+", self.graph.nodes[str(self.current_node)]['name'])
                host = match.group() if match else None
                print(f"Found target node '{target_name}' at node {current_node}")
                return host, current_node

            # Add predecessors to the stack for further search
            predecessors = list(self.graph.predecessors(current_node))
            stack.extend(predecessors)

        #print(f"No node with name '{target_name}' found.")
        return None, None

    def extract_cve_from_node_name(self, node_name):
        match = re.search(r'(CVE-\d{4}-\d{4,7})', node_name, re.IGNORECASE)
        return match.group(1) if match else None

    def find_node_ids_by_names(self, target_names: list):
        return [self.name_to_id_map.get(name) for name in target_names]


    def apply_action(self, action_id, **kwargs):
        """Dispatcher for all defensive actions."""
        if action_id == 0:
            return self._action_do_nothing()
        elif action_id == 1:
            return self._action_network_analysis(**kwargs)
        elif action_id == 2:
            return self._action_network_filtering(**kwargs)
        elif action_id == 3:
            return self._action_restore_software(**kwargs)
        # NEW: Added Action 4
        elif action_id == 4:
            return self._action_restore_network_connection(**kwargs)
        else:
            return {"status": "error", "message": f"Invalid action ID: {action_id}"}


    # This section Refers to the Agents Actions and how these affect the environment 

    # ACTION 0
    def _action_do_nothing(self):

        """Action 0: Does nothing. Serves as a baseline."""
        print("Action applied: Do Nothing.")

        return {"status": "success", "action": "Do Nothing", "changes": 0}

    # # ACTION 1 
    def _action_network_filtering(self, source_ip, dest_ip):

        print(f"Action applied: Network Filtering between {source_ip} and {dest_ip}.")
        analysis_result = self._action_network_analysis(source_ip=source_ip, dest_ip=dest_ip)
        paths_to_block = analysis_result.get("potential_paths", [])
        
        if not paths_to_block:
            return {"status": "failure", "message": f"No direct reachability path found from {source_ip} to {dest_ip}."}
        
        changes_count = 0

        for path in paths_to_block:
            u, v = path['from_node_id'], path['to_node_id']

            if self.graph.has_edge(u, v):
                edge_data = self.graph.get_edge_data(u, v)
                self.removed_edges.append((u, v, edge_data))
                
                self.graph.remove_edge(u, v)
                changes_count += 1
                print(f"  -> Removed edge from node {u} to {v}. Stored for potential restoration.")

        if changes_count > 0:
            self._update_graph_state() 

        return {"status": "success", "action": "Network Filtering", "changes": changes_count}
    
    # ACTION 2
    # MODIFIED: Now updates state after completion
    def _action_restore_software(self, ip_address, cve_id=None):

        print(f"Action applied: Restore Software on {ip_address}" + (f" for {cve_id}" if cve_id else "."))
        nodes_to_remove_ids = []

        if cve_id:
            node_name = f"{cve_id}_[{ip_address}]"
            node_id = self.name_to_id_map.get(node_name)
            if node_id: nodes_to_remove_ids.append(node_id)
        else:
            names_to_find = [f"HostCompromise_[{ip_address}]", f"Privs_[{ip_address}]_Root", f"Privs_[{ip_address}]_User"]
            found_ids = [self.name_to_id_map.get(name) for name in names_to_find]
            nodes_to_remove_ids.extend([nid for nid in found_ids if nid is not None])

        if not nodes_to_remove_ids:
            return {"status": "failure", "message": f"No relevant nodes found for host {ip_address}."}

        removed_count = 0

        for node_id in nodes_to_remove_ids:
            if self.graph.has_node(node_id):
                self.graph.remove_node(node_id)
                removed_count += 1
        
        if removed_count > 0:
            self._update_graph_state()

        return {"status": "success", "action": "Restore Software", "changes": removed_count}
    
    # ACTION 3
    def _action_restore_network_connection(self):
        """Action 4: Restores the most recently removed network edge."""

        print("Action applied: Restore Network Connection.")
        if not self.removed_edges:
            return {"status": "failure", "message": "No removed edges to restore."}

        u, v, edge_data = self.removed_edges.pop()
        
        self.graph.add_edge(u, v, **edge_data)
        print(f"  -> Restored edge from node {u} to {v}.")
        
        self._update_graph_state() 
        return {"status": "success", "action": "Restore Network Connection", "changes": 1, "restored_edge": (u, v)}
