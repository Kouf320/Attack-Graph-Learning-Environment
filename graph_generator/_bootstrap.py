"""
Bootstrap helper for the graph_generator package.

The ViolenceLang attack-graph generator depends on a specific version of
``mal-toolbox`` (0.0.21) whose API differs from the current PyPI release
(``specification.load_language_specification_from_mar``,
``classes_factory.LanguageClassesFactory`` ...).  That version is *not*
installable from PyPI, so the package ships a vendored, self-contained copy of
the library under ``graph_generator/vendor/maltoolbox``.

Rather than force a ``pip install`` of the vendored package, this module
prepends the vendor directory to ``sys.path`` at import time so the generator
"just works" from a fresh checkout.  The original location inside
``mal/ViolenceLang/mal-toolbox`` is kept only as a fallback, so deleting the
``mal/`` folder does not break anything once the vendored copy is in place.

It also exposes :data:`REPO_ROOT` and a few well-known paths so the rest of the
package never hard-codes absolute locations.
"""

from __future__ import annotations

import os
import sys
import warnings

# graph_generator/ -> repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "graph_generator")

# Self-contained vendored mal-toolbox (preferred).  This is a directory that
# *contains* the ``maltoolbox`` package, so it is what goes on ``sys.path``.
VENDOR_DIR = os.path.join(PACKAGE_DIR, "vendor")

# Legacy location, used only if the vendored copy is absent.
LEGACY_MALTOOLBOX = os.path.join(REPO_ROOT, "mal", "ViolenceLang", "mal-toolbox")

# Default ViolenceLang MAL language specification (.mar) shipped with the tool.
DEFAULT_LANG_MAR = os.path.join(
    PACKAGE_DIR, "lang", "org.mal-lang.Violencelang-1.0.0.mar"
)

# Where the RL environment looks for ready-to-use attack graphs.
ATTACK_GRAPHS_DIR = os.path.join(REPO_ROOT, "attack_graphs")

# Where the RL environment's CVSS/CVE lookup database lives.
DEFAULT_DB_PATH = os.path.join(
    REPO_ROOT, "database", "vulnerability-remediation-database.db"
)

# Where this tool stores topology specification files.
TOPOLOGIES_DIR = os.path.join(PACKAGE_DIR, "topologies")


def ensure_maltoolbox_on_path() -> str:
    """Prepend the vendored mal-toolbox to ``sys.path`` (idempotent).

    Prefers the self-contained ``graph_generator/vendor`` copy; falls back to
    the legacy ``mal/ViolenceLang/mal-toolbox`` location only if the vendored
    copy is missing.  Returns the path that was wired in, or raises
    ``RuntimeError`` with an actionable message if neither is available.
    """
    if os.path.isdir(os.path.join(VENDOR_DIR, "maltoolbox")):
        chosen = VENDOR_DIR
    elif os.path.isdir(os.path.join(LEGACY_MALTOOLBOX, "maltoolbox")):
        chosen = LEGACY_MALTOOLBOX
    else:
        raise RuntimeError(
            "mal-toolbox not found. Expected the vendored copy at:\n  "
            f"{os.path.join(VENDOR_DIR, 'maltoolbox')}\n"
            "(or the legacy copy under mal/ViolenceLang/mal-toolbox). The "
            "ViolenceLang generator cannot run without it."
        )
    if chosen not in sys.path:
        sys.path.insert(0, chosen)
    return chosen


def silence_pjs_warnings() -> None:
    """python_jsonschema_objects is chatty about JSON-schema versions.

    These warnings are harmless for our use, so collapse them to keep CLI and
    server output readable.
    """
    warnings.filterwarnings(
        "ignore",
        message="Schema version not specified.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Schema id not specified.*",
        category=UserWarning,
    )
