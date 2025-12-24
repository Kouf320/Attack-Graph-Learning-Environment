# RewardModel.py

import logging

class RewardModel:

    def get_reward(self, **kwargs):
        
        raise NotImplementedError("This method should be overridden by subclasses.")



class DefaultRewardModel(RewardModel):
    """
    - A terminal reward that incentivizes efficiency (fewer vulnerabilities met).
    - A conditional intermediate reward that applies different formulas based
      on the structural position of a CVE in the attack graph.
    """
    def __init__(self, norm_factor_attempt=21.147181, norm_factor_cia=17.8572, non_action_value=-0.1):
        self.NORM_F_ATTEMPT = norm_factor_attempt
        self.NORM_F_CIA = norm_factor_cia
        self.NON_ACTION_VALUE = 0

    def get_reward(self, **kwargs):

        if kwargs.get('is_terminal', False):
            vuln_met = kwargs.get('vuln_met', 0)
            total_vuln = kwargs.get('total_vuln', 0)
            if total_vuln == 0:
                return 0  
            return 100 * (1 - vuln_met / total_vuln)

        cvss_string = kwargs.get('cvss_string', 'Empty')
        node_position = kwargs.get('node_position')

        if not cvss_string or cvss_string == 'Empty' or node_position is None:
            return self.NON_ACTION_VALUE

        attack_metric = self._parse_cvss(cvss_string)
        if not attack_metric:
            return self.NON_ACTION_VALUE

        if node_position == 0:
            value = round(attack_metric.get("AV", 0) * attack_metric.get("AC", 0) * attack_metric.get("PR", 0) * attack_metric.get("UI", 0) * self.NORM_F_ATTEMPT, 3)
            logging.info(f"Applying attacker-effort reward model. Value: {value}")
            return value
        elif node_position == 1:
            value = round((1 - attack_metric.get('C', 0)) * (1 - attack_metric.get('I', 0)) * (1 - attack_metric.get('A', 0)) * self.NORM_F_CIA, 3)
            logging.info(f"Applying CIA-impact reward model. Value: {value}")
            return value
        else:
            return self.NON_ACTION_VALUE

    def is_vulnerability_met(self, graph, node_id):
        """A vulnerability is met if the node is a CVE."""
        node_name = graph.nodes[str(node_id)]['name']
        return node_name.startswith("CVE")

    def _parse_cvss(self, cvss_string):
        """Parses a CVSS string into a dictionary of quantitative values."""
        vector = {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.20}
        complexity = {'L': 0.77, 'H': 0.44}
        privileges = {'N': 0.85, 'L': 0.62, 'H': 0.27}
        interaction = {'N': 0.85, 'R': 0.62}
        cia = {'N': 0.00, 'L': 0.22, 'H': 0.56}

        if cvss_string.startswith("CVSS:3.1/"):
            cvss_string = cvss_string[len("CVSS:3.1/"):]

        metric_pairs = cvss_string.split('/')
        parsed_qn_metrics = {}
        for pair in metric_pairs:
            try:
                metric, value = pair.split(':')
                if metric == 'AV':
                    parsed_qn_metrics['AV'] = vector.get(value)
                elif metric == "AC":
                    parsed_qn_metrics['AC'] = complexity.get(value)
                elif metric == "PR":
                    parsed_qn_metrics['PR'] = privileges.get(value)
                elif metric == "UI":
                    parsed_qn_metrics['UI'] = interaction.get(value)
                elif metric in ["C", "I", "A"]:
                    parsed_qn_metrics[metric] = cia.get(value)
            except ValueError:
                logging.warning(f"Could not parse CVSS metric pair: {pair}")
                continue
        return parsed_qn_metrics