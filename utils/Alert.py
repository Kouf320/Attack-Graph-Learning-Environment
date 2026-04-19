import random
import time
import numpy as np
import json
from datetime import datetime
import random


# Function to generate a Suricata-like alert
import random
from datetime import datetime

class AlertGenerator:
    def __init__(self):
        self.proto = "TCP"
        self.in_iface = "eth0"
        self.default_app_proto = "http"
        self.choices = [1, 2, 3, 4, 5]
        self.weights = [0.40, 0.20, 0.20, 0.10, 0.10]  # Adjust probabilities as needed


    def _generate_common_fields(self, src_ip, src_port, dest_ip, dest_port, category, signature, severity = None):
        
        if severity is None:
            # fallback for prev behavior if no severity is provided
            severity = random.choices(self.choices, weights=self.weights, k=1)[0]

        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f+0000"),
            "event_type": "alert",
            "src_ip": src_ip,
            "src_port": src_port,
            "dest_ip": dest_ip,
            "dest_port": dest_port,
            "proto": self.proto,
            "alert": {
                "action": "allowed",
                "gid": 1,
                "signature_id": np.random.randint(1000000, 9999999),
                "rev": 1,
                "signature": signature,
                "category": category,
                "severity": severity
            },
            "flow_id": np.random.randint(100000000000000, 999999999999999),
            "in_iface": self.in_iface,
            "payload": "Base64-encoded payload string",
            "payload_printable": "ASCII representation of the payload",
            "stream": 1,
            "app_proto": self.default_app_proto,
            "flow": {
                "pkts_toserver": np.random.randint(1, 10),
                "pkts_toclient": np.random.randint(1, 10),
                "bytes_toserver": np.random.randint(100, 1000),
                "bytes_toclient": np.random.randint(100, 1000),
                "start": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f+0000")
            }
        }

    def generate_alert_recon(self, src_ip, src_port, dest_ip, dest_port, severity = 3):
        alert = self._generate_common_fields(
            src_ip, src_port, dest_ip, dest_port,
            category="Potential Network Scan",
            signature="ET SCAN Possible Nmap Scan Detected",
            severity = np.random.randint(2, 3)
        )
        alert["http"] = {
            "hostname": "example.com",
            "url": "/malicious/path",
            "http_user_agent": "Mozilla/5.0",
            "http_method": "GET",
            "protocol": "HTTP/1.1",
            "status": 200,
            "length": np.random.randint(50, 500)
        }
        return alert

    def generate_alert_local(self, src_ip, src_port, dest_ip, dest_port, severity = 5):
        alert = self._generate_common_fields(
            src_ip, src_port, dest_ip, dest_port,
            category="Local Exploitation Attempt",
            signature="ET EXPLOIT Local Privilege Escalation Attempt",
            severity = np.random.randint(3,5)
        )
        alert["exploit_details"] = {
            "exploit_type": "Local Privilege Escalation",
            "tool_used": "Example Exploit Tool",
            "success": random.choice([True, False])
        }
        return alert

    def generate_alert_network(self, src_ip, src_port, dest_ip, dest_port, severity = 5):
        alert = self._generate_common_fields(
            src_ip, src_port, dest_ip, dest_port,
            category="Network Exploitation Attempt",
            signature="ET EXPLOIT Network Exploitation Attempt Detected",
            severity = np.random.randint(3, 5)
        )
        alert["network_details"] = {
            "target_service": "SSH",
            "vulnerability": "CVE-2024-XXXX",
            "exploit_method": "Buffer Overflow"
        }
        return alert


    def generate_alert_noise(self):
        # simulate a benign but suspicious event from a random internal host
        src_ip = f"192.168.4.{np.random.randint(50, 100)}" 
        dest_ip = "8.8.8.8" 
        return self._generate_common_fields(
            src_ip, np.random.randint(10000, 60000), dest_ip, 53,
            category="Policy Violation",
            signature="POLICY Internal Host DNS Query to External Server",
            severity=2 # Low severity
        )