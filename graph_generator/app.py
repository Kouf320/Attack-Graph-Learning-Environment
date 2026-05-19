"""
Flask backend for the ViolenceLang Topology Builder.

Serves the single-page topology-builder UI and exposes a small JSON API that
turns a topology specification into an RL-ready attack graph using the
integrated generator, writing the result straight into ``attack_graphs/`` so it
is immediately usable by the reinforcement-learning environment.

Run
---
    python -m graph_generator.app           # then open http://127.0.0.1:5000
    # or:  ./graph_generator/run.sh

The server is local-only by default (binds 127.0.0.1).  Nothing leaves the
machine; all generation happens in-process with the bundled mal-toolbox.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import traceback

from flask import Flask, jsonify, request, send_from_directory

from . import _bootstrap
from .violence_generator import generate_attack_graph, normalize_topology
from .rl_bridge import validate_cves_against_db

_bootstrap.silence_pjs_warnings()

WEB_DIR = os.path.join(_bootstrap.PACKAGE_DIR, "web")

# Topology spec search locations (read), in priority order.  Each entry is
# only used if it exists, so extra paths are harmless.  The legacy
# ``mal/ViolenceLang/ViolenceGenerator`` folder is included as an optional
# source so its specs remain discoverable *if* the folder is still present;
# the canonical specs have been copied into ``graph_generator/topologies`` so
# deleting ``mal/`` loses nothing.
TOPOLOGY_SOURCES = [
    _bootstrap.TOPOLOGIES_DIR,
    os.path.join(_bootstrap.REPO_ROOT, "mal", "ViolenceLang", "ViolenceGenerator"),
]

app = Flask(__name__, static_folder=None)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str, default: str = "topology") -> str:
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    stem = _SAFE_NAME.sub("_", stem).strip("_")
    return stem or default


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/web/<path:filename>")
def web_assets(filename):
    return send_from_directory(WEB_DIR, filename)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "repo_root": _bootstrap.REPO_ROOT,
            "lang_mar": _bootstrap.DEFAULT_LANG_MAR,
            "lang_mar_exists": os.path.exists(_bootstrap.DEFAULT_LANG_MAR),
            "attack_graphs_dir": _bootstrap.ATTACK_GRAPHS_DIR,
            "db_path": _bootstrap.DEFAULT_DB_PATH,
            "db_exists": os.path.exists(_bootstrap.DEFAULT_DB_PATH),
        }
    )


@app.route("/api/topologies")
def list_topologies():
    """List available topology spec files across known source folders."""
    items = []
    seen = set()
    for src in TOPOLOGY_SOURCES:
        if not os.path.isdir(src):
            continue
        for fn in sorted(os.listdir(src)):
            if not fn.endswith(".json"):
                continue
            full = os.path.join(src, fn)
            if full in seen:
                continue
            seen.add(full)
            # Only surface files that look like topologies.
            try:
                with open(full, "r") as fh:
                    head = json.load(fh)
                if isinstance(head, dict) and (
                    "onlineHosts" in head or "hosts" in head
                ):
                    n_hosts = len(head.get("onlineHosts", head.get("hosts", [])))
                    items.append(
                        {
                            "name": fn,
                            "path": full,
                            "source": os.path.relpath(src, _bootstrap.REPO_ROOT),
                            "hosts": n_hosts,
                        }
                    )
            except (ValueError, OSError):
                continue
    return jsonify({"topologies": items})


@app.route("/api/topology")
def load_topology():
    """Return a topology file, normalised to the canonical onlineHosts shape."""
    path = request.args.get("path", "")
    # security: only allow files inside known source folders
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(s)) for s in TOPOLOGY_SOURCES):
        return jsonify({"error": "Path not allowed."}), 403
    if not os.path.isfile(real):
        return jsonify({"error": "File not found."}), 404
    with open(real, "r") as fh:
        raw = json.load(fh)
    return jsonify({"raw": raw, "normalized": normalize_topology(raw)})


@app.route("/api/graphs")
def list_graphs():
    """List attack-graph JSONs already present in attack_graphs/."""
    d = _bootstrap.ATTACK_GRAPHS_DIR
    items = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                full = os.path.join(d, fn)
                try:
                    with open(full, "r") as fh:
                        g = json.load(fh)
                    items.append(
                        {
                            "name": fn,
                            "assets": len(g.get("assets", {})),
                            "associations": len(g.get("associations", [])),
                        }
                    )
                except (ValueError, OSError):
                    items.append({"name": fn, "assets": "?", "associations": "?"})
    return jsonify({"graphs": items})


@app.route("/api/validate-cves", methods=["POST"])
def validate_cves():
    payload = request.get_json(force=True, silent=True) or {}
    topology = payload.get("topology")
    if topology is None:
        return jsonify({"error": "Missing 'topology'."}), 400
    try:
        report = validate_cves_against_db(topology)
        return jsonify(report)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate an RL-ready attack graph from a posted topology.

    Body: ``{"topology": {...}, "name": "my_scenario", "save_topology": true}``

    Writes:
      * the attack graph to ``attack_graphs/<name>.json``
      * (optionally) the topology spec to ``graph_generator/topologies/<name>.json``
    """
    payload = request.get_json(force=True, silent=True) or {}
    topology = payload.get("topology")
    if topology is None:
        return jsonify({"error": "Missing 'topology'."}), 400

    stem = _safe_stem(payload.get("name", "topology"))
    save_topology = bool(payload.get("save_topology", True))

    graph_path = os.path.join(_bootstrap.ATTACK_GRAPHS_DIR, f"{stem}.json")
    topo_path = os.path.join(_bootstrap.TOPOLOGIES_DIR, f"{stem}.json")

    try:
        os.makedirs(_bootstrap.ATTACK_GRAPHS_DIR, exist_ok=True)
        if save_topology:
            os.makedirs(_bootstrap.TOPOLOGIES_DIR, exist_ok=True)
            with open(topo_path, "w") as fh:
                json.dump(topology, fh, indent=2)

        model_name = (
            f"ViolenceLang Model :: {stem} "
            f"({_dt.datetime.now().strftime('%Y-%m-%d %H:%M')})"
        )
        graph = generate_attack_graph(
            topology, output_path=graph_path, model_name=model_name, verbose=False
        )

        # Best-effort RL compliance summary (does not require torch).
        access_nodes = [
            k for k, v in graph.get("assets", {}).items()
            if v.get("metaconcept") == "Access"
        ]
        recon_nodes = [
            k for k, v in graph.get("assets", {}).items()
            if str(v.get("name", "")).startswith("Reconnaissance")
        ]

        cve_report = validate_cves_against_db(topology)

        return jsonify(
            {
                "ok": True,
                "graph_path": graph_path,
                "graph_rel": os.path.relpath(graph_path, _bootstrap.REPO_ROOT),
                "topology_path": topo_path if save_topology else None,
                "assets": len(graph.get("assets", {})),
                "associations": len(graph.get("associations", [])),
                "attackers": len(graph.get("attackers", {})),
                "access_goal_candidates": len(access_nodes),
                "recon_entry_points": len(recon_nodes),
                "cve_summary": {
                    "ok": len(cve_report["ok"]),
                    "empty_cvss": len(cve_report["empty_cvss"]),
                    "missing": len(cve_report["missing"]),
                    "db_available": cve_report["db_available"],
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


def main():
    host = os.environ.get("VIOLENCE_HOST", "127.0.0.1")
    port = int(os.environ.get("VIOLENCE_PORT", "5000"))
    print("=" * 70)
    print(" ViolenceLang Topology Builder")
    print("=" * 70)
    print(f"  Repo root        : {_bootstrap.REPO_ROOT}")
    print(f"  Language (.mar)  : {_bootstrap.DEFAULT_LANG_MAR}")
    print(f"  Attack graphs -> : {_bootstrap.ATTACK_GRAPHS_DIR}")
    print(f"  CVSS database    : {_bootstrap.DEFAULT_DB_PATH}")
    print(f"\n  Open:  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()
