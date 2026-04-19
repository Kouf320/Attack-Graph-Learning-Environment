import math
from collections import Counter

class StreamingEntropy:

    def __init__(self):
        self.counts = Counter()
        self.total_alerts = 0

    def update(self, batch):
        if not batch: return self.calculate_entropy()
        self.total_alerts += len(batch)
        self.counts.update(batch)

        return self.calculate_entropy()

    def calculate_entropy(self):

        if self.total_alerts == 0: return 0.0

        entropy = 0.0

        for count in self.counts.values():
            probability = count / self.total_alerts
            entropy -= probability * math.log2(probability)

        return entropy

def bin_risk_score(score):
    """Convert a numerical risk score into a category"""
    if score >= 0.9:

        return 'Critical'

    elif score >= 0.75:

        return 'High'

    elif score >= 0.5:

        return 'Medium'

    else:

        return 'Low'