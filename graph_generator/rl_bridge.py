"""
Bridge between ViolenceLang topologies and the RL environment.

These helpers let you go straight from a topology specification to a live
``GraphEnvironment`` (or a Gymnasium wrapper) without manually shuttling JSON
files around — though they will also write the intermediate attack-graph JSON to
``attack_graphs/`` if you ask, so it can be reused later exactly like the
hand-made graphs.

Also provides :func:`validate_cves_against_db`, which checks whether the CVEs in
a topology are present (with a non-empty CVSS string) in the RL environment's
remediation database.  This matters because ``GraphEnvironment`` recomputes edge
rewards by looking each CVE up in that database — a CVE that is missing or has an
empty CVSS string contributes no severity-based weight to its edges.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Union

from . import _bootstrap
from .violence_generator import generate_attack_graph, normalize_topology, _load_topology


def topology_to_graph_json(
    topology: Union[str, dict],
    output_path: Optional[str] = None,
    lang_mar_path: Optional[str] = None,
    model_name: str = "ViolenceLang Model (integrated generator)",
) -> dict:
    """Convert a topology into the RL-ready attack-graph dict (optionally saved)."""
    return generate_attack_graph(
        topology,
        output_path=output_path,
        lang_mar_path=lang_mar_path,
        model_name=model_name,
        verbose=False,
    )


def build_environment_from_topology(
    topology: Union[str, dict],
    goal_node=None,
    config_path: Optional[str] = None,
    save_to: Optional[str] = None,
    lang_mar_path: Optional[str] = None,
):
    """Build a :class:`environment.graph_env.GraphEnvironment` from a topology.

    Parameters
    ----------
    topology
        Topology dict or path.
    goal_node
        Optional explicit goal-node id; otherwise the env auto-detects the
        deepest ``Access`` node (its default behaviour).
    config_path
        Path to ``config.json`` (defaults to the repo's).
    save_to
        If given, the generated attack-graph JSON is also written here so it can
        be re-used later just like a hand-authored graph.
    lang_mar_path
        Optional override for the ViolenceLang ``.mar``.
    """
    import sys

    if _bootstrap.REPO_ROOT not in sys.path:
        sys.path.insert(0, _bootstrap.REPO_ROOT)
    from environment.graph_env import GraphEnvironment  # noqa: E402

    graph_json = topology_to_graph_json(
        topology, output_path=save_to, lang_mar_path=lang_mar_path
    )
    config_path = config_path or os.path.join(_bootstrap.REPO_ROOT, "config.json")
    return GraphEnvironment(graph_json, goal_node=goal_node, config_path=config_path)


def build_gym_env_from_topology(
    topology: Union[str, dict],
    perspective: str = "attacker",
    goal_node=None,
    config_path: Optional[str] = None,
    save_to: Optional[str] = None,
    lang_mar_path: Optional[str] = None,
    **gym_kwargs,
):
    """Build a Gymnasium env (``NetworkAttackEnv`` / ``NetworkDefenderEnv``).

    ``perspective`` is ``"attacker"`` or ``"defender"``.
    """
    import sys

    if _bootstrap.REPO_ROOT not in sys.path:
        sys.path.insert(0, _bootstrap.REPO_ROOT)
    from environment.gym_env import NetworkAttackEnv, NetworkDefenderEnv  # noqa: E402

    graph_json = topology_to_graph_json(
        topology, output_path=save_to, lang_mar_path=lang_mar_path
    )
    config_path = config_path or os.path.join(_bootstrap.REPO_ROOT, "config.json")
    cls = NetworkAttackEnv if perspective == "attacker" else NetworkDefenderEnv
    return cls(graph_json, goal_node=goal_node, config_path=config_path, **gym_kwargs)


# --------------------------------------------------------------------------- #
# CVE / CVSS database validation
# --------------------------------------------------------------------------- #
def validate_cves_against_db(
    topology: Union[str, dict], db_path: Optional[str] = None
) -> dict:
    """Check each CVE in the topology against the RL environment's database.

    Returns a report dict::

        {
          "db_path": "...",
          "db_available": bool,
          "results": [
            {"cve": "...", "in_db": bool, "has_cvss": bool, "cvss_string": "..."},
            ...
          ],
          "missing": [...],           # CVEs not present at all
          "empty_cvss": [...],        # present but blank CVSS string
          "ok": [...],                # present with usable CVSS string
        }
    """
    db_path = db_path or _bootstrap.DEFAULT_DB_PATH
    data = normalize_topology(_load_topology(topology))

    # collect unique CVEs
    cves = []
    seen = set()
    for host in data.get("onlineHosts", []):
        for vuln in host.get("vulnerabilities", []) or []:
            cve = vuln.get("cve")
            if cve and cve not in seen:
                seen.add(cve)
                cves.append(cve)

    report = {
        "db_path": db_path,
        "db_available": os.path.exists(db_path),
        "results": [],
        "missing": [],
        "empty_cvss": [],
        "ok": [],
    }
    if not report["db_available"]:
        for cve in cves:
            report["results"].append(
                {"cve": cve, "in_db": False, "has_cvss": False, "cvss_string": None}
            )
            report["missing"].append(cve)
        return report

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for cve in cves:
            cur.execute("SELECT cvss_string FROM vulnerability WHERE cve = ?", (cve,))
            row = cur.fetchone()
            if row is None:
                report["results"].append(
                    {"cve": cve, "in_db": False, "has_cvss": False, "cvss_string": None}
                )
                report["missing"].append(cve)
            else:
                cvss_string = row[0]
                has_cvss = bool(cvss_string and str(cvss_string).strip())
                report["results"].append(
                    {
                        "cve": cve,
                        "in_db": True,
                        "has_cvss": has_cvss,
                        "cvss_string": cvss_string,
                    }
                )
                (report["ok"] if has_cvss else report["empty_cvss"]).append(cve)
    finally:
        conn.close()
    return report
