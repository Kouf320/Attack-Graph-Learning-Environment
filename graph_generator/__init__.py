"""
graph_generator
===============

Plug-and-play bridge between ViolenceLang topology specifications and the
Attack-Graph reinforcement-learning environment.

Two ways to use it
-------------------
1. Keep using ready-made attack-graph JSON files exactly as before — nothing in
   the RL environment changes.
2. *Author your own* attack graphs from a high-level topology spec (hosts,
   their CVEs/CVSS vectors, and the connections between them) and turn them into
   RL-ready attack graphs with one call / one click.

Public API
----------
- :func:`graph_generator.violence_generator.generate_attack_graph`
- :func:`graph_generator.rl_bridge.build_environment_from_topology`
- :func:`graph_generator.rl_bridge.build_gym_env_from_topology`
"""

from . import _bootstrap  # noqa: F401  (ensures bundled mal-toolbox is on path)

__all__ = ["violence_generator", "rl_bridge", "_bootstrap"]
