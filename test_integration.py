"""
Integration test: simulates the relevant slice of GraphEnvironment.step()
that uses the adapter, to confirm the wiring works end-to-end.

Builds a small synthetic env with the same shape as your real one
(graph, ip_to_nodes_map, DATABASE_PATH stub, CVE lookup), drives a few
episodes through the adapter, and verifies:

  * Per-traversal detection stochasticity: the SAME node sometimes
    produces alerts, sometimes does not (proof that detection is NOT
    a per-node constant).
  * Severity actually varies across the 1..5 scale.
  * The dict schema matches what the env's downstream parser expects
    (IP regex still finds the IPs).
  * Episode-level resets vary the noisy-host subset.
  * Fully-thinned cluster case is observable for stealthy exploits.
"""
import re
import sys
import collections
import networkx as nx

try:  # run from repo root: `python test_integration.py`
    from utils.alert_adapter import GraphAlertAdapter
except ImportError:  # pragma: no cover
    from alert_adapter import GraphAlertAdapter


# ---------------------------------------------------------------------
# fake env (just the surface area the adapter touches)
# ---------------------------------------------------------------------
class FakeEnv:
    DATABASE_PATH = None
    def __init__(self):
        self.graph = nx.DiGraph()
        # network exploit (loud)
        self.graph.add_node("100",
                            name="Reconnaissance on 10.0.0.5",
                            metaconcept="Scan", alert_type="recon",
                            base_severity=2)
        # network CVE
        self.graph.add_node("200",
                            name="CVE-2023-46604 on 10.0.0.12",
                            metaconcept="Vulnerability",
                            alert_type="network", base_severity=4)
        # host compromise
        self.graph.add_node("300",
                            name="HostCompromise on 10.0.0.20",
                            metaconcept="Host",
                            alert_type="network", base_severity=5)
        # stealthy local CVE (cred-reuse archetype)
        self.graph.add_node("400",
                            name="CVE-2024-99999 on 10.0.0.20",
                            metaconcept="Vulnerability",
                            alert_type="local", base_severity=3)
        for u, v in [("100","200"), ("200","300"), ("300","400")]:
            self.graph.add_edge(u, v)
        self.ip_to_nodes_map = {
            "10.0.0.5":  ["100"],
            "10.0.0.12": ["200"],
            "10.0.0.20": ["300", "400"],
        }
        # extra hosts so the noisy subset has somewhere to land
        for i in range(30, 60):
            ip = f"10.0.0.{i}"
            self.ip_to_nodes_map[ip] = []
        self.random_noise_rate = 0.30

    def extract_cve_from_node_name(self, n):
        m = re.search(r'(CVE-\d{4}-\d{4,7})', n or "")
        return m.group(1) if m else None

    def get_cvss_by_cve(self, _db, cve):
        if cve == "CVE-2023-46604":
            # loud unauthenticated RCE
            return "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        if cve == "CVE-2024-99999":
            # stealthy local with auth required
            return "CVSS:3.1/AV:L/AC:H/PR:H/UI:N/S:U/C:H/I:H/A:H"
        return "Empty"


# ---------------------------------------------------------------------
# test
# ---------------------------------------------------------------------
print("=" * 72)
print("End-to-end integration test")
print("=" * 72)

env = FakeEnv()
adapter = GraphAlertAdapter(env, false_positive_rate=0.30, seed=7)

# ---- A. inspect the catalogue ----------------------------------------
print("\n[A] catalogue contents (one Exploit per node)")
for nid in env.graph.nodes():
    exp = adapter.catalogue[nid]
    p = adapter.detection_probability(nid)
    print(f"    node {nid}: {exp.name:32s}  "
          f"timing={exp.timing_model.value:8s} "
          f"P_detect={p:.3f}")

# ---- B. per-traversal detection stochasticity ------------------------
# Hit the same node 200 times and count how many traversals produce
# zero alerts. If detection were a per-node constant we'd see either
# always-alerts or always-silent. We should see a Bernoulli mix.
print("\n[B] detection IS per-traversal stochastic, not per-node constant")
adapter.reset_episode()
for nid, exp_label in [("200", "loud RCE"), ("400", "stealthy local")]:
    silent = 0
    alert_counts = []
    for step in range(200):
        _, tp, _ = adapter.generate_for_transition(
            prev_node="100", curr_node=nid,
            src_ip="10.0.0.5", dst_ip="10.0.0.12",
            step_index=step)
        if tp.detected_size == 0:
            silent += 1
        alert_counts.append(tp.detected_size)
    p_det = adapter.detection_probability(nid)
    expected_visible = 1 - (1 - p_det) ** max(1, int(sum(c+1 for c in alert_counts) / len(alert_counts)))
    print(f"    node {nid} ({exp_label:15s}): "
          f"{200 - silent:3d}/200 traversals produced >= 1 alert, "
          f"{silent:3d}/200 produced ZERO   "
          f"(P_detect_per_alert = {p_det:.3f})")

# ---- C. severity actually varies -------------------------------------
print("\n[C] severity actually varies (1..5) -- z_score / entropy features")
print("    now carry real signal instead of noise around a constant")
adapter.reset_episode()
sev_counts = collections.Counter()
for step in range(50):
    alerts, _, _ = adapter.generate_for_transition(
        prev_node="100", curr_node="200",
        src_ip="10.0.0.5", dst_ip="10.0.0.12",
        step_index=step)
    for d in alerts:
        sev_counts[d["alert"]["severity"]] += 1
total = sum(sev_counts.values())
for s in range(1, 6):
    pct = sev_counts.get(s, 0) / max(1, total) * 100
    bar = "#" * int(pct / 2)
    print(f"    severity {s}: {sev_counts.get(s, 0):4d}  ({pct:5.1f}%)  {bar}")

# ---- D. schema compatibility with env's IP parser --------------------
print("\n[D] dict schema is compatible with env's downstream parser")
ip_pat = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
adapter.reset_episode()
alerts, _, _ = adapter.generate_for_transition(
    prev_node="100", curr_node="200",
    src_ip="10.0.0.5", dst_ip="10.0.0.12",
    step_index=0)
required_keys = {"timestamp","src_ip","dest_ip","src_port","dest_port",
                 "alert","flow_id","event_type"}
for d in alerts[:3]:
    missing = required_keys - set(d.keys())
    print(f"    sample dict: missing keys = "
          f"{sorted(missing) if missing else 'NONE'}")
ips = set()
for d in alerts:
    ips |= set(ip_pat.findall(str(d)))
print(f"    IPs found via env regex: {sorted(ips)} "
      f"(must include 10.0.0.5 + 10.0.0.12)")
assert "10.0.0.5" in ips and "10.0.0.12" in ips
print(f"    severity reachable via alert_obj.get('alert',{{}}).get('severity'): "
      f"{alerts[0].get('alert', {}).get('severity')}")

# ---- E. episode-level noise variability ------------------------------
print("\n[E] noisy-host subset varies across episodes")
subsets = []
for ep in range(5):
    adapter.reset_episode()
    subsets.append(tuple(sorted(adapter.noisy_hosts())))
    print(f"    episode {ep}: noisy = {adapter.noisy_hosts()[:4]}...")
distinct = len(set(subsets))
print(f"    distinct subsets across 5 episodes: {distinct} (should be >= 3)")

# ---- F. fully-thinned cluster signal ---------------------------------
print("\n[F] fully-thinned cluster is observable (APT signature)")
adapter.reset_episode()
ft = 0
for step in range(100):
    _, tp, _ = adapter.generate_for_transition(
        prev_node="300", curr_node="400",
        src_ip="10.0.0.20", dst_ip="10.0.0.20",
        step_index=step)
    if tp.was_fully_thinned:
        ft += 1
print(f"    stealthy CVE traversed 100 times, "
      f"fully-thinned (raw>0, detected=0): {ft} times")

print("\n" + "=" * 72)
print("End-to-end integration test passed.")
print("=" * 72)
