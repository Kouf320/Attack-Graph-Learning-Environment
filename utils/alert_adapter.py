"""
alert_adapter.py
================

Bridge between the new four-layer `AlertGenerator` and your existing
`GraphEnvironment`'s alert-consumer code, which expects Suricata-style
dicts of the form:

    {
        "timestamp": "...",
        "event_type": "alert",
        "src_ip": "...", "src_port": 12345,
        "dest_ip": "...", "dest_port": 80,
        "proto": "TCP",
        "alert": {
            "action": "allowed",
            "gid": 1,
            "signature_id": 1234567,
            "rev": 1,
            "signature": "...",
            "category": "...",
            "severity": 3,
        },
        "flow_id": ...,
        "in_iface": "eth0",
        "payload": "...",
        "payload_printable": "...",
        "stream": 1,
        "app_proto": "http",
        "flow": {...},
        # optionally http / exploit_details / network_details
    }

This is the exact schema produced by `utils.Alert.AlertGenerator`. We
keep it verbatim so your downstream code (line 691 onwards in
`GraphEnvironment.step()`):

    for alert_obj in self.current_alert_group:
        alert_str = str(alert_obj)
        found_ips = ip_pattern.findall(alert_str)
        ...
        severity = alert_obj.get('alert', {}).get('severity', 1)

continues to work without any change.

The adapter also exposes a tiny `_generate_alerts_for_transition` that
the modified `step()` can call as a one-liner.

Dependencies: numpy + the alert_generator.py module.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

try:  # project-relative imports; fall back to flat imports for standalone runs
    from utils.alert_generator import (
        AlertGenerator,
        AlertGroup,
        Alert,
        Exploit,
        TimingModel,
    )
    from utils.exploit_catalogue import build_exploit_catalogue
except ImportError:  # pragma: no cover
    from alert_generator import (
        AlertGenerator,
        AlertGroup,
        Alert,
        Exploit,
        TimingModel,
    )
    from exploit_catalogue import build_exploit_catalogue


# =====================================================================
# Single-alert flattening
# =====================================================================

# The base UTC epoch we use when converting the generator's float
# timestamps (seconds since episode start) to ISO strings. Reset on
# every reset_episode() so timestamps walk forward consistently.
_EPOCH = datetime.utcnow()


def _to_suricata_dict(a: Alert,
                      episode_epoch: datetime,
                      proto: str = "TCP",
                      in_iface: str = "eth0",
                      app_proto: str = "http") -> Dict[str, Any]:
    """
    Convert one Alert into the Suricata-style dict your env consumes.
    Adds the bookkeeping fields (timestamp, flow_id, payload, ...) that
    the old `utils.Alert.AlertGenerator` was producing, so the schema
    is byte-compatible with what `GraphEnvironment.step()` already parses.
    """
    ts = (episode_epoch + timedelta(seconds=float(a.time))) \
        .strftime("%Y-%m-%dT%H:%M:%S.%f+0000")

    d: Dict[str, Any] = {
        "timestamp": ts,
        "event_type": "alert",
        "src_ip": a.src_ip,
        "src_port": int(a.src_port),
        "dest_ip": a.dst_ip,
        "dest_port": int(a.dst_port),
        "proto": proto,
        "alert": {
            "action": "allowed",
            "gid": 1,
            "signature_id": int(np.random.randint(1_000_000, 9_999_999)),
            "rev": 1,
            "signature": a.signature,
            "category": a.category,
            "severity": int(a.severity),
        },
        "flow_id": int(np.random.randint(10**14, 10**15)),
        "in_iface": in_iface,
        "payload": "Base64-encoded payload string",
        "payload_printable": "ASCII representation of the payload",
        "stream": 1,
        "app_proto": app_proto,
        "flow": {
            "pkts_toserver": int(np.random.randint(1, 10)),
            "pkts_toclient": int(np.random.randint(1, 10)),
            "bytes_toserver": int(np.random.randint(100, 1000)),
            "bytes_toclient": int(np.random.randint(100, 1000)),
            "start": ts,
        },
        # extra metadata so debugging / reward shaping can inspect
        # which layer-3 verdict produced this alert
        "_meta": {
            "is_true_positive": bool(a.is_true_positive),
            "exploit_name": a.exploit_name,
        },
    }

    # category-specific sub-blocks (mirrors the old AlertGenerator's
    # generate_alert_recon / _local / _network variants)
    cat = (a.category or "").lower()
    if "scan" in cat or "recon" in cat:
        d["http"] = {
            "hostname": "example.com",
            "url": "/probe",
            "http_user_agent": "Mozilla/5.0",
            "http_method": "GET",
            "protocol": "HTTP/1.1",
            "status": 200,
            "length": int(np.random.randint(50, 500)),
        }
    elif "local" in cat or "priv" in cat:
        d["exploit_details"] = {
            "exploit_type": "Local Privilege Escalation",
            "tool_used": "n/a",
            "success": True,
        }
    elif "network" in cat or "exploit" in cat:
        d["network_details"] = {
            "target_service": "tcp/" + str(a.dst_port),
            "vulnerability": a.exploit_name or "unspecified",
            "exploit_method": "n/a",
        }
    # policy violations (FPs) get no extra block

    return d


# =====================================================================
# The adapter class
# =====================================================================

class GraphAlertAdapter:
    """
    Owns the `AlertGenerator` and the `{node_id: Exploit}` catalogue,
    and exposes the methods the `GraphEnvironment` needs:

        adapter.reset_episode(false_positive_rate=...)
            -> call from env.reset_recon() / env.reset()

        adapter.generate_for_transition(prev_node, curr_node,
                                        src_ip, dst_ip, step_index)
            -> returns (list_of_suricata_dicts, tp_group, fp_group)
            list_of_suricata_dicts is the value to assign to
            env.current_alert_group.

        adapter.generate_idle_step(step_index)
            -> returns (list_of_suricata_dicts, None, fp_group)
            use when the attacker did not act this step.

    Typical wiring inside `GraphEnvironment.__init__`:

        from alert_adapter import GraphAlertAdapter
        ...
        self.alert_adapter = GraphAlertAdapter(
            env=self,
            false_positive_rate=self.random_noise_rate,
            seed=42,
        )

    And in `step()` (replacing lines 558-681):

        src_ip, dst_ip = source_ip_for_alert, dest_ip_for_alert
        self.current_alert_group, tp, fp = \
            self.alert_adapter.generate_for_transition(
                prev_node=self.previous_node,
                curr_node=self.current_node,
                src_ip=src_ip, dst_ip=dst_ip,
                step_index=self._current_step,
            )
        # NOTE: if tp.was_fully_thinned, the attacker acted but
        # produced no alerts -- you may want to log that explicitly.

    Parameters
    ----------
    env : GraphEnvironment
        Used to read the graph and resolve host IPs. Stored by ref.
    false_positive_rate : float
        Mirrors env.random_noise_rate. Pass it through; the adapter
        owns the live value going forward.
    seed : Optional[int]
        Seed for the inner generator's RNG.
    catalogue : Optional[dict[str, Exploit]]
        Pre-built `{node_id: Exploit}` map. If None, we build one via
        `build_exploit_catalogue(env)`.
    generator_kwargs : Optional[dict]
        Extra kwargs forwarded to AlertGenerator (e.g. `hawkes_alpha`,
        `enable_thinning`, `step_duration`).
    """

    def __init__(
        self,
        env,
        false_positive_rate: float,
        *,
        seed: Optional[int] = None,
        catalogue: Optional[Dict[str, Exploit]] = None,
        generator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env = env

        # Build (or accept) the per-node exploit catalogue
        self.catalogue: Dict[str, Exploit] = (
            catalogue if catalogue is not None
            else build_exploit_catalogue(env))

        # The host list is the set of IPs the env already discovered
        # during `_map_ips_to_nodes`. We pass it as-is so Layer 4's
        # noisy-host subset is drawn over real topology IPs.
        if not hasattr(env, "ip_to_nodes_map"):
            raise RuntimeError(
                "GraphAlertAdapter: env has no ip_to_nodes_map. "
                "Make sure _map_ips_to_nodes() has run before "
                "constructing the adapter.")
        hosts = sorted(env.ip_to_nodes_map.keys())
        if not hosts:
            # Some attack graphs name nodes "Host 1" rather than by IP, so the
            # env's IP regex finds nothing and ip_to_nodes_map is empty. Fall
            # back to node ids as host tokens so the generator can still build;
            # the env's alert->node attribution already degrades gracefully in
            # this case (get_valid_action_mask falls back to Do-Nothing).
            hosts = [str(n) for n in env.graph.nodes()]

        # Build the generator
        gk = dict(generator_kwargs or {})
        # sensible defaults for an RL-environment use case
        gk.setdefault("step_duration", 1.0)
        gk.setdefault("hawkes_mu", 0.3)
        gk.setdefault("hawkes_alpha", 1.2)
        gk.setdefault("hawkes_beta", 1.5)

        self.generator = AlertGenerator(
            hosts=hosts,
            exploits={e.name: e for e in self.catalogue.values()},
            false_positive_rate=float(false_positive_rate),
            seed=seed,
            **gk,
        )

        # Map from node_id to the exploit NAME used in the generator
        # (the generator keys on exploit name, the catalogue keys on
        # node id, so we need the lookup)
        self._node_to_exploit_name: Dict[str, str] = {
            nid: exp.name for nid, exp in self.catalogue.items()
        }

        # Episode-local clock anchor for ISO timestamps
        self._episode_epoch = datetime.utcnow()

        # Track the last AlertGroup pair (for diagnostics / reward shaping)
        self.last_tp_group: Optional[AlertGroup] = None
        self.last_fp_group: Optional[AlertGroup] = None

    # ---- episode lifecycle ------------------------------------------
    def reset_episode(self,
                      false_positive_rate: Optional[float] = None) -> None:
        """
        Call once per episode (from env.reset_recon / env.reset /
        env.reset_curriculum). Re-draws the noisy-host subset, clears
        Hawkes history, and resets the timestamp epoch.
        """
        self.generator.reset_episode(false_positive_rate)
        self._episode_epoch = datetime.utcnow()
        self.last_tp_group = None
        self.last_fp_group = None

    def set_false_positive_rate(self, rate: float) -> None:
        """Change the FP rate live (also re-draws the noisy subset)."""
        self.generator.set_false_positive_rate(rate)

    # ---- main per-step entry points ---------------------------------
    def generate_for_transition(
        self,
        prev_node: Any,
        curr_node: Any,
        src_ip: str,
        dst_ip: str,
        step_index: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[AlertGroup], AlertGroup]:
        """
        Generate one step's alerts for an attacker transition.

        The exploit is selected by the **destination** node (the node
        the attacker just arrived at), matching the convention in your
        current `step()`: `current_node_data.get('detection_prob', ...)`
        keys on the current/landed node.

        Returns
        -------
        (alert_dicts, tp_group, fp_group)
            alert_dicts : list of Suricata-style dicts to assign to
                          env.current_alert_group.
            tp_group    : the raw AlertGroup for the true-positive
                          cluster (may have detected_size=0 -- check
                          tp_group.was_fully_thinned to detect
                          fully-stealthy steps).
            fp_group    : the false-positive AlertGroup for this step.
        """
        curr_node = str(curr_node)
        exploit_name = self._node_to_exploit_name.get(curr_node)
        if exploit_name is None:
            # node missing from catalogue (e.g. dynamically added):
            # treat as an idle step and just emit FPs
            return self.generate_idle_step(step_index)

        tp, fp = self.generator.step(
            step_index=step_index,
            attacker_exploit=exploit_name,
            src_ip=str(src_ip),
            dst_ip=str(dst_ip),
        )
        self.last_tp_group, self.last_fp_group = tp, fp

        alerts = []
        if tp is not None:
            alerts.extend(_to_suricata_dict(a, self._episode_epoch)
                          for a in tp.alerts)
        alerts.extend(_to_suricata_dict(a, self._episode_epoch)
                      for a in fp.alerts)
        return alerts, tp, fp

    def generate_idle_step(
        self,
        step_index: int,
    ) -> Tuple[List[Dict[str, Any]], None, AlertGroup]:
        """Generate one step of FPs only (attacker idle)."""
        _, fp = self.generator.step(step_index=step_index)
        self.last_tp_group, self.last_fp_group = None, fp
        alerts = [_to_suricata_dict(a, self._episode_epoch)
                  for a in fp.alerts]
        return alerts, None, fp

    # ---- diagnostics ------------------------------------------------
    def detection_probability(self, node_id: Any) -> float:
        """P_detect for the exploit attached to a graph node."""
        name = self._node_to_exploit_name.get(str(node_id))
        if name is None:
            return 0.0
        return self.generator.detection_probability(name)

    def noisy_hosts(self) -> List[str]:
        """Current chronically-noisy host subset (varies per episode)."""
        return self.generator.noisy_hosts()


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    """
    Smoke-test the adapter using the same fake env we used for the
    catalogue builder. Verifies:
      1. The adapter constructs without touching the real env.
      2. generate_for_transition returns Suricata-shaped dicts.
      3. The dicts contain the IPs we passed in (so the env's IP regex
         parsing will see them).
      4. Episode lifecycle (reset_episode) varies the noisy subset.
    """
    import networkx as _nx
    import re as _re

    class _FakeEnv:
        DATABASE_PATH = None
        def __init__(self):
            self.graph = _nx.DiGraph()
            self.graph.add_node("1", name="Reconnaissance on 10.0.0.5",
                                metaconcept="Scan", alert_type="recon",
                                base_severity=2)
            self.graph.add_node("2", name="CVE-2023-46604 on 10.0.0.12",
                                metaconcept="Vulnerability",
                                alert_type="network", base_severity=4)
            self.graph.add_node("3", name="HostCompromise on 10.0.0.20",
                                metaconcept="Host",
                                alert_type="network", base_severity=5)
            self.graph.add_edge("1", "2")
            self.graph.add_edge("2", "3")
            self.ip_to_nodes_map = {
                "10.0.0.5":  ["1"],
                "10.0.0.12": ["2"],
                "10.0.0.20": ["3"],
            }

        def extract_cve_from_node_name(self, n):
            m = _re.search(r'(CVE-\d{4}-\d{4,7})', n or "")
            return m.group(1) if m else None

        def get_cvss_by_cve(self, _db, cve):
            if cve == "CVE-2023-46604":
                return "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
            return "Empty"

    print("=" * 64)
    print("GraphAlertAdapter smoke test")
    print("=" * 64)

    env = _FakeEnv()
    adapter = GraphAlertAdapter(env, false_positive_rate=0.30, seed=42)

    print(f"\nCatalogue covers {len(adapter.catalogue)} nodes")
    for nid in env.graph.nodes():
        p = adapter.detection_probability(nid)
        exp = adapter.catalogue[nid]
        print(f"  node {nid} -> {exp.name:35s}"
              f"  P_detect={p:.3f}  timing={exp.timing_model.value}")

    print(f"\nNoisy hosts at episode 0: {adapter.noisy_hosts()}")

    # Step 0: attacker moves to node 2 (the CVE-2023-46604 RCE)
    adapter.reset_episode()
    alerts, tp, fp = adapter.generate_for_transition(
        prev_node="1", curr_node="2",
        src_ip="10.0.0.5", dst_ip="10.0.0.12",
        step_index=0,
    )
    print(f"\nStep 0 (attacker exploits CVE on 10.0.0.12):")
    print(f"  raw cluster size = {tp.raw_cluster_size}")
    print(f"  detected size    = {tp.detected_size}")
    print(f"  FP alerts        = {len(fp.alerts)}")
    print(f"  total dicts      = {len(alerts)}")
    print(f"  first TP dict (truncated):")
    if alerts:
        a0 = alerts[0]
        print(f"    timestamp = {a0['timestamp']}")
        print(f"    src_ip    = {a0['src_ip']}:{a0['src_port']}")
        print(f"    dest_ip   = {a0['dest_ip']}:{a0['dest_port']}")
        print(f"    alert.signature = {a0['alert']['signature']}")
        print(f"    alert.category  = {a0['alert']['category']}")
        print(f"    alert.severity  = {a0['alert']['severity']}")
        print(f"    _meta          = {a0['_meta']}")

    # Confirm the dicts are byte-compatible with the env's IP parser
    ip_pat = _re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
    found = set()
    for d in alerts:
        for ip in ip_pat.findall(str(d)):
            found.add(ip)
    print(f"\n  IPs found in dicts via the env's regex: {sorted(found)}")
    assert "10.0.0.12" in found, "destination IP missing -- env parser would break"

    # Verify episode-to-episode noise variability
    subsets = set()
    for _ in range(4):
        adapter.reset_episode()
        subsets.add(tuple(adapter.noisy_hosts()))
    print(f"\nDistinct noisy-host subsets over 4 episode resets: {len(subsets)}")

    print("\n" + "=" * 64)
    print("Adapter self-test passed.")
    print("=" * 64)
