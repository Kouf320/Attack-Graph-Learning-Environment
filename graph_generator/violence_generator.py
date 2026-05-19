"""
ViolenceLang attack-graph generator (integrated, importable + CLI).

This is a refactor of the original
``mal/ViolenceLang/ViolenceGenerator/violence_model_generator.py`` into a clean,
reusable module.  The *exact* CVSS -> MAL-asset mapping rules of the original are
preserved (see :func:`build_model_from_topology`); only the plumbing changed:

  * the host-iteration logic is wrapped in a function that takes a topology
    *dict* (or path) and returns the RL-ready model *dict*;
  * input topologies in either the modern (``onlineHosts`` /
    ``expectedConnections``) or the legacy per-host (``hosts`` /
    ``connections``) shape are accepted and normalised;
  * the bundled mal-toolbox is wired onto ``sys.path`` automatically;
  * Neo4j ingestion is optional and lazily imported;
  * a small ``argparse`` CLI is provided.

The produced JSON has the structure ``{metadata, assets, associations,
attackers}`` — identical to what ``GraphEnvironment.initialize_from_json``
consumes (``attack_graphs/ag.json``).  So the output is plug-and-play with the
reinforcement-learning environment.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Union

from . import _bootstrap

_bootstrap.silence_pjs_warnings()
_bootstrap.ensure_maltoolbox_on_path()

# Imported *after* the bundled mal-toolbox is on sys.path.
from maltoolbox.language import classes_factory  # noqa: E402
from maltoolbox.language import specification  # noqa: E402
from maltoolbox.model import model as malmodel  # noqa: E402

from .cvss_utils import parse_cvss  # noqa: E402


class _Counter:
    """Monotonic integer id generator (callable)."""

    def __init__(self, start: int = 0):
        self.count = start

    def __call__(self) -> int:
        current = self.count
        self.count += 1
        return current


# --------------------------------------------------------------------------- #
# Topology normalisation
# --------------------------------------------------------------------------- #
def normalize_topology(topology: dict) -> dict:
    """Return a topology in canonical ``{onlineHosts, expectedConnections}`` form.

    The generator's native format (e.g. ``sl300_big.json``) uses::

        {"onlineHosts": [{id, hostname, ip[], os, status, group, vulnerabilities[]}],
         "expectedConnections": [{"source": id, "destination": id}]}

    Some older specs (e.g. ``example_topology.json``) instead use ``hosts`` with
    a per-host ``connections`` list.  Both are accepted; this normaliser maps the
    legacy shape onto the canonical one so the rest of the pipeline only deals
    with one layout.
    """
    if "onlineHosts" in topology:
        hosts = topology.get("onlineHosts", [])
        conns = list(topology.get("expectedConnections", []))
        # Also fold in any per-host connections, just in case.
        for h in hosts:
            for c in h.get("connections", []) or []:
                conns.append(c)
        return {"onlineHosts": hosts, "expectedConnections": conns}

    # Legacy "hosts" / per-host "connections" layout.
    hosts = topology.get("hosts", [])
    conns = []
    for h in hosts:
        for c in h.get("connections", []) or []:
            conns.append({"source": c.get("source"), "destination": c.get("destination")})
    # de-dup connections
    seen = set()
    deduped = []
    for c in conns:
        key = (c.get("source"), c.get("destination"))
        if key not in seen and None not in key:
            seen.add(key)
            deduped.append(c)
    return {"onlineHosts": hosts, "expectedConnections": deduped}


def _load_topology(topology: Union[str, dict]) -> dict:
    if isinstance(topology, str):
        with open(topology, "r") as fh:
            topology = json.load(fh)
    if not isinstance(topology, dict):
        raise TypeError("topology must be a dict or a path to a JSON file")
    return normalize_topology(topology)


# --------------------------------------------------------------------------- #
# Core model construction (logic preserved from the original generator)
# --------------------------------------------------------------------------- #
def build_model_from_topology(
    topology: Union[str, dict],
    lang_mar_path: Optional[str] = None,
    model_name: str = "ViolenceLang Model (integrated generator)",
    verbose: bool = False,
):
    """Build and return a populated mal-toolbox ``Model`` from a topology.

    Parameters
    ----------
    topology
        Topology dict or path to a topology JSON file.
    lang_mar_path
        Path to the ViolenceLang ``.mar`` specification.  Defaults to the copy
        bundled with this package.
    model_name
        Human-readable name stored in the model metadata.
    verbose
        Print per-asset creation logs (mirrors the original script's output).

    Returns
    -------
    maltoolbox.model.model.Model
        The populated model.  Call ``.save_to_file(path)`` or use
        :func:`model_to_dict` to obtain the RL-ready JSON.
    """
    lang_mar_path = lang_mar_path or _bootstrap.DEFAULT_LANG_MAR
    if not os.path.exists(lang_mar_path):
        raise FileNotFoundError(f"Language .mar not found: {lang_mar_path}")

    data = _load_topology(topology)

    lang_spec = specification.load_language_specification_from_mar(lang_mar_path)
    factory = classes_factory.LanguageClassesFactory(lang_spec)
    factory.create_classes()

    model = malmodel.Model(model_name, lang_spec, factory)
    id_counter = _Counter()
    added_association_signatures: set = set()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    def add_association_if_new(assoc_name: str, links: dict) -> bool:
        """De-duplicating association adder (preserved from the original)."""
        sig_parts = []
        for role, assets_list in links.items():
            if not assets_list or any(a is None for a in assets_list):
                return False
            eids = []
            for a in assets_list:
                if hasattr(a, "eid") and a.eid is not None:
                    eids.append(a.eid)
                else:
                    log(f"ERROR: asset in role '{role}' for '{assoc_name}' has no EID.")
                    return False
            sig_parts.append((role, frozenset(eids)))
        signature = (assoc_name, frozenset(sig_parts))
        if signature in added_association_signatures:
            return False
        constructor = getattr(factory.ns, assoc_name, None)
        if constructor is None:
            log(f"ERROR: association metaconcept '{assoc_name}' not found.")
            return False
        model.add_association(constructor(**links))
        added_association_signatures.add(signature)
        return True

    online_hosts = data.get("onlineHosts", [])
    expected_connections = data.get("expectedConnections", [])
    host_assets_map: dict = {}

    log(f"Processing {len(online_hosts)} hosts...")

    # ------------------------------------------------------------------ #
    # Per-host asset + association construction
    # (CVSS -> asset mapping rules are byte-for-byte the original logic)
    # ------------------------------------------------------------------ #
    for host_data in online_hosts:
        host_id = str(host_data.get("id"))
        if not host_id or host_id == "None":
            log(f"Skipping host with missing id: {host_data.get('hostname', '?')}")
            continue
        ip_addresses = host_data.get("ip", [])
        primary_ip = ip_addresses[0] if ip_addresses else "NoIP"
        host_assets_map[host_id] = {"PrimaryIP": primary_ip}
        vulns = host_data.get("vulnerabilities", [])

        scan = local_x = network_x = adjacent_x = None
        root_priv = user_priv = host_comp = None
        internet = denial = unsuccessful = chain = None

        if vulns:
            scan = factory.ns.Scan(name=f"Reconnaissance_on_[{primary_ip}]")
            scan.eid = str(id_counter())
            model.add_asset(scan)
            host_assets_map[host_id]["Scan"] = scan

        for vuln in vulns:
            cve_id = vuln.get("cve", f"UnknownCVE_{id_counter()}")
            cvss_string = vuln.get("cvss", None)
            vuln_asset = factory.ns.Vulnerability(name=f"{cve_id}_[{primary_ip}]")
            vuln_asset.eid = str(id_counter())
            model.add_asset(vuln_asset)
            if scan:
                add_association_if_new("ReconInit", {"scans": [scan], "vulns": [vuln_asset]})
            if not cvss_string:
                continue

            m = parse_cvss(cvss_string)
            av = m.get("Attack Vector")
            ac = m.get("Attack Complexity")
            c_conf = m.get("Confidentiality")
            i_integ = m.get("Integrity")
            a_avail = m.get("Availability")

            # --- exploit assets keyed on Attack Vector ---
            if av == "L":
                if not local_x:
                    local_x = factory.ns.Local(name=f"LocalExploit_[{primary_ip}]")
                    local_x.eid = str(id_counter())
                    model.add_asset(local_x)
                    host_assets_map[host_id]["Local"] = local_x
                add_association_if_new("AvL", {"vulns": [vuln_asset], "lexploits": [local_x]})
            elif av == "N":
                if not network_x:
                    network_x = factory.ns.Network(name=f"NetworkExploit_[{primary_ip}]")
                    network_x.eid = str(id_counter())
                    model.add_asset(network_x)
                    host_assets_map[host_id]["Network"] = network_x
                add_association_if_new("AvN", {"vulns": [vuln_asset], "nexploits": [network_x]})
                if not internet:
                    internet = factory.ns.Internet(name=f"InternetAccessFor_[{primary_ip}]")
                    internet.eid = str(id_counter())
                    model.add_asset(internet)
                    host_assets_map[host_id]["Internet"] = internet
                add_association_if_new(
                    "HasInternet", {"globalconnections": [internet], "nexploits": [network_x]}
                )
            elif av == "A":
                if not adjacent_x:
                    adjacent_x = factory.ns.Adjacent(name=f"AdjacentExploit_[{primary_ip}]")
                    adjacent_x.eid = str(id_counter())
                    model.add_asset(adjacent_x)
                    host_assets_map[host_id]["Adjacent"] = adjacent_x
                add_association_if_new("AvA", {"vulns": [vuln_asset], "aexploits": [adjacent_x]})

            # --- outcomes of Local exploit ---
            if local_x:
                if c_conf == "H":
                    if not root_priv:
                        root_priv = factory.ns.Privileges(name=f"Root_on_Host_[{primary_ip}]")
                        root_priv.eid = str(id_counter())
                        model.add_asset(root_priv)
                        host_assets_map[host_id]["RootPrivileges"] = root_priv
                    add_association_if_new("ViHLN", {"lexploits": [local_x], "privesc": [root_priv]})
                if ac == "H":
                    if not unsuccessful:
                        unsuccessful = factory.ns.Unsuccesfull(
                            name=f"InsufficientAttack_[{primary_ip}]_AC_H"
                        )
                        unsuccessful.eid = str(id_counter())
                        model.add_asset(unsuccessful)
                        host_assets_map[host_id]["Unsuccesfull"] = unsuccessful
                    add_association_if_new("AcHl", {"lexploits": [local_x], "insufs": [unsuccessful]})
                if a_avail in ("H", "L"):
                    if not denial:
                        denial = factory.ns.Denial(
                            name=f"DenialOfService_[{primary_ip}]_A_{a_avail}"
                        )
                        denial.eid = str(id_counter())
                        model.add_asset(denial)
                        host_assets_map[host_id]["Denial"] = denial
                    add_association_if_new("VaHL", {"lexploits": [local_x], "ddosattacks": [denial]})

            # --- outcomes of Network exploit ---
            if network_x:
                if c_conf == "H":
                    if not root_priv:
                        root_priv = factory.ns.Privileges(name=f"Root_on_Host_[{primary_ip}]")
                        root_priv.eid = str(id_counter())
                        model.add_asset(root_priv)
                        host_assets_map[host_id]["RootPrivileges"] = root_priv
                    add_association_if_new("ViHN", {"nexploits": [network_x], "privesc": [root_priv]})
                if (i_integ in ("H", "L")) and not (c_conf == "H" and root_priv):
                    if not user_priv:
                        user_priv = factory.ns.Privileges(
                            name=f"User_Privilege_Escalation_on_[{primary_ip}]"
                        )
                        user_priv.eid = str(id_counter())
                        model.add_asset(user_priv)
                        host_assets_map[host_id]["UserPrivileges"] = user_priv
                    add_association_if_new("ViHN", {"nexploits": [network_x], "privesc": [user_priv]})

            # --- outcomes of Adjacent exploit ---
            if adjacent_x:
                if i_integ == "H":
                    if not root_priv:
                        root_priv = factory.ns.Privileges(name=f"Root_on_Host_[{primary_ip}]")
                        root_priv.eid = str(id_counter())
                        model.add_asset(root_priv)
                        host_assets_map[host_id]["RootPrivileges"] = root_priv
                    add_association_if_new("VcHA", {"aexploits": [adjacent_x], "privesc": [root_priv]})

            # --- Host compromise from root privileges (any path) ---
            if root_priv:
                if not host_comp:
                    host_comp = factory.ns.Host(name=f"Host_Compromise_[{primary_ip}]")
                    host_comp.eid = str(id_counter())
                    model.add_asset(host_comp)
                    host_assets_map[host_id]["Host"] = host_comp
                add_association_if_new(
                    "Standard", {"privesc": [root_priv], "compromises": [host_comp]}
                )

            # --- Insufficient attack -> complex (chain) attack -> root ---
            if unsuccessful:
                if not chain:
                    chain = factory.ns.Chain(
                        name=f"ComplexAttack_[{primary_ip}]_after_Insufficient"
                    )
                    chain.eid = str(id_counter())
                    model.add_asset(chain)
                    host_assets_map[host_id]["Chain"] = chain
                add_association_if_new("Complex", {"insufs": [unsuccessful], "mulattacks": [chain]})
                if chain and root_priv:
                    add_association_if_new("multistage", {"scans": [chain], "privesc": [root_priv]})

    # ------------------------------------------------------------------ #
    # Inter-host reachability (expectedConnections)
    # ------------------------------------------------------------------ #
    log(f"Processing {len(expected_connections)} expected connections...")
    for conn in expected_connections:
        src_id = str(conn.get("source"))
        dst_id = str(conn.get("destination"))
        src = host_assets_map.get(src_id)
        dst = host_assets_map.get(dst_id)
        if not src or not dst:
            log(f"  Skipping connection {src_id}->{dst_id}: host not found in map.")
            continue
        src_comp = src.get("Host")
        dst_scan = dst.get("Scan")
        dst_adj = dst.get("Adjacent")
        src_ip = src.get("PrimaryIP", "UnknownSrcIP")
        dst_ip = dst.get("PrimaryIP", "UnknownDestIP")

        if src_comp and (dst_scan or dst_adj):
            access = factory.ns.Access(name=f"Access_to_{dst_ip}_from_{src_ip}")
            access.eid = str(id_counter())
            model.add_asset(access)
            add_association_if_new(
                "Reachability", {"compromises": [src_comp], "accesses": [access]}
            )
            if dst_scan:
                add_association_if_new(
                    "DoReconOnReachableHost", {"accesses": [access], "scans": [dst_scan]}
                )
            if dst_adj:
                add_association_if_new(
                    "CanExploit", {"accesses": [access], "aexploits": [dst_adj]}
                )

    # ------------------------------------------------------------------ #
    # Attacker entry point (first asset, 'Scan' attack step) — as original.
    # ------------------------------------------------------------------ #
    if model.assets:
        attacker = malmodel.Attacker()
        attacker.entry_points = [(model.assets[0], ["Scan"])]
        model.add_attacker(attacker)

    return model


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def model_to_dict(model) -> dict:
    """Serialise a populated model into the RL-ready ``{metadata, assets,
    associations, attackers}`` dict, without leaving a temp file behind.
    """
    if hasattr(model, "_to_dict"):
        return model._to_dict()
    # Fall back to save/reload (mal-toolbox 0.0.21 path).
    import tempfile

    with tempfile.NamedTemporaryFile("r+", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_to_file(tmp_path)
        with open(tmp_path, "r") as fh:
            return json.load(fh)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def generate_attack_graph(
    topology: Union[str, dict],
    output_path: Optional[str] = None,
    lang_mar_path: Optional[str] = None,
    model_name: str = "ViolenceLang Model (integrated generator)",
    verbose: bool = False,
) -> dict:
    """End-to-end: topology -> RL-ready attack-graph dict (optionally saved).

    Returns the model dict.  If ``output_path`` is given, the dict is also
    written there as pretty JSON (this is what you point the RL env at).
    """
    model = build_model_from_topology(
        topology, lang_mar_path=lang_mar_path, model_name=model_name, verbose=verbose
    )
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        model.save_to_file(output_path)
        with open(output_path, "r") as fh:
            return json.load(fh)
    return model_to_dict(model)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="violence_generator",
        description=(
            "Generate an RL-ready attack graph from a ViolenceLang topology "
            "specification.  Output JSON matches the format consumed by "
            "GraphEnvironment (attack_graphs/ag.json)."
        ),
    )
    p.add_argument("topology", help="Path to the input topology JSON file.")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output attack-graph JSON path. Defaults to attack_graphs/<topology_stem>.json",
    )
    p.add_argument(
        "-l", "--lang", default=None,
        help="Path to the ViolenceLang .mar (defaults to the bundled copy).",
    )
    p.add_argument("-n", "--name", default="ViolenceLang Model (integrated generator)",
                   help="Model name stored in metadata.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress per-asset logs.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.topology))[0]
        output = os.path.join(_bootstrap.ATTACK_GRAPHS_DIR, f"{stem}.json")
    result = generate_attack_graph(
        args.topology,
        output_path=output,
        lang_mar_path=args.lang,
        model_name=args.name,
        verbose=not args.quiet,
    )
    print(
        f"\nDone. Wrote {len(result.get('assets', {}))} assets and "
        f"{len(result.get('associations', []))} associations to:\n  {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
