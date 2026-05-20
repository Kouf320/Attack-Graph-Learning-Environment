"""
alert_generator.py
==================

A realistic IDS alert-stream generator implementing the four-layer
generative pipeline described in the companion tutorial
("Why Realistic IDS Alert Streams Are Hard to Simulate").

The four layers, each independently toggleable:

  1. Cluster generation   (metadata axis)
       A single attacker exploit emits a *cluster* of correlated alerts,
       not a single token. Cluster size ~ LogNormal(mu_e, sigma_e).

  2. Per-exploit TIMING   (temporal axis)
       Each exploit names its own temporal model. NOT every exploit is
       a Hawkes process:
         - HAWKES   : self-exciting cascade   (port scans, worm-like spread)
         - POISSON  : memoryless background   (generic noise, some recon)
         - PERIODIC : near-fixed cadence      (brute-force, C2 beacons)
         - BURST    : one tight clump         (single-shot RCE exploits)
       Forcing one temporal model on every exploit is the same mistake
       as a constant per-node detection probability -- just dressed up.

  3. Detection thinning   (detection axis)
       Each alert in a cluster survives with probability P_detect(e),
       a logistic function of the exploit's CVSS / ATT&CK features.
       Stealthy exploits are mostly invisible; noisy ones are not.
       Detection is drawn fresh per traversal -- NOT a per-node constant
       -- so the agent cannot memorise an alert/no-alert table.

  4. False-positive mixture (spatial axis)
       Benign false alerts are NOT spread uniformly. They concentrate on
       a small "noisy host" subset (Zipf-weighted) with a thin uniform
       tail, reproducing the operational Gini ~ 0.7. The noisy subset is
       fixed within an episode and re-drawn across episodes.

The false-positive *rate* is a first-class constructor argument
(`false_positive_rate`) and can be changed at run time via
`set_false_positive_rate()` or `reset_episode(false_positive_rate=...)`.

Dependencies: numpy only.

Author: companion code to the alert-realism tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Dict, List, Tuple
import numpy as np


# =====================================================================
# Timing models
# =====================================================================

class TimingModel(str, Enum):
    """
    The temporal model an exploit's alert cluster follows.

    HAWKES   -- self-exciting cascade. One alert raises the chance of
                the next; produces tight bursts with heavy-tailed gaps.
                Good for: port scans, scanning worms, noisy lateral
                movement.

    POISSON  -- memoryless. Alerts arrive at a roughly constant rate
                with exponential gaps; no clustering beyond chance.
                Good for: generic background activity, some benign recon.

    PERIODIC -- near-fixed cadence. Alerts arrive at a base interval
                plus bounded jitter; UNDER-dispersed (more regular than
                Poisson), the opposite of a burst.
                Good for: brute-force attempts, C2 beacons, automated
                credential stuffing.

    BURST    -- one tight clump then silence. All alerts land within a
                short window with little internal structure.
                Good for: single-shot RCE exploits, one-and-done actions.
    """
    HAWKES = "hawkes"
    POISSON = "poisson"
    PERIODIC = "periodic"
    BURST = "burst"


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class Exploit:
    """
    Description of a single attacker exploit / technique.

    Fields drive three of the four layers:
      * (mu_cluster, sigma_cluster) parameterise the LogNormal cluster
        size in Layer 1.
      * timing_model (+ its parameters) selects the Layer-2 model.
      * (av, ac, auth, stealth) feed the Layer-3 logistic detector.

    Parameters
    ----------
    name : str
        Human-readable identifier, e.g. "CVE-2023-46604" or
        "node_142:recon".
    av : float in [0, 1]
        CVSS Attack Vector score. ~1.0 for Network, ~0.0 for Local.
    ac : float in [0, 1]
        CVSS Attack Complexity score.
    auth : int in {0, 1}
        1 if the exploit requires authentication / privileges (quieter
        on the wire -- looks like legitimate traffic).
    stealth : float in [0, 1]
        Per-technique stealth score (e.g. from a MITRE ATT&CK lookup).
        Higher = better at evading detection.
    mu_cluster : float
        LogNormal mu for cluster size. Median size ~ exp(mu_cluster).
    sigma_cluster : float
        LogNormal sigma for cluster size. Larger = heavier right tail.
    timing_model : TimingModel
        Which Layer-2 temporal model this exploit's cluster follows.
    timing_params : dict
        Optional per-model timing parameters. Recognised keys:
          HAWKES   : 'mu', 'alpha', 'beta'  (override generator defaults
                     for this exploit only)
          POISSON  : 'rate'                 (events per second)
          PERIODIC : 'period', 'jitter'     (base interval + uniform
                     jitter half-width, both in seconds)
          BURST    : 'span'                 (total window the clump
                     occupies, in seconds)
    signature_weights : Optional[dict[str, float]]
        Categorical distribution over signature IDs for alerts in this
        exploit's cluster.
    category : str
        Suricata-style category string attached to every alert.
    severity_weights : Optional[Sequence[float]]
        Categorical distribution over severity levels 1..5.
    """
    name: str
    av: float = 1.0
    ac: float = 0.5
    auth: int = 0
    stealth: float = 0.0
    mu_cluster: float = 2.8
    sigma_cluster: float = 0.6
    timing_model: TimingModel = TimingModel.HAWKES
    timing_params: Dict[str, float] = field(default_factory=dict)
    signature_weights: Optional[Dict[str, float]] = None
    category: str = "Generic Exploitation Attempt"
    severity_weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        for fld, val in (("av", self.av), ("ac", self.ac),
                         ("stealth", self.stealth)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Exploit '{self.name}': {fld}={val} must be in [0, 1]."
                )
        if self.auth not in (0, 1):
            raise ValueError(
                f"Exploit '{self.name}': auth must be 0 or 1, got {self.auth}."
            )
        if self.sigma_cluster <= 0:
            raise ValueError(
                f"Exploit '{self.name}': sigma_cluster must be > 0."
            )
        # accept plain strings for timing_model
        if not isinstance(self.timing_model, TimingModel):
            self.timing_model = TimingModel(str(self.timing_model).lower())
        if self.signature_weights:
            total = float(sum(self.signature_weights.values()))
            if total <= 0:
                raise ValueError(
                    f"Exploit '{self.name}': signature_weights sum <= 0."
                )
            self.signature_weights = {
                k: v / total for k, v in self.signature_weights.items()
            }
        if self.severity_weights is not None:
            sw = np.asarray(self.severity_weights, dtype=float)
            if np.any(sw < 0) or sw.sum() <= 0:
                raise ValueError(
                    f"Exploit '{self.name}': severity_weights invalid."
                )
            self.severity_weights = (sw / sw.sum()).tolist()


@dataclass
class Alert:
    """A single IDS alert record."""
    time: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    signature: str
    category: str
    severity: int
    is_true_positive: bool
    exploit_name: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "signature": self.signature,
            "category": self.category,
            "severity": self.severity,
            "is_true_positive": self.is_true_positive,
            "exploit_name": self.exploit_name,
        }


@dataclass
class AlertGroup:
    """A cluster of alerts from one attacker action OR one step's FP bundle."""
    alerts: List[Alert]
    step: int
    raw_cluster_size: int
    detected_size: int
    kind: str
    exploit_name: Optional[str] = None
    timing_model: Optional[str] = None

    def signature_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for a in self.alerts:
            hist[a.signature] = hist.get(a.signature, 0) + 1
        return hist

    def severity_histogram(self) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for a in self.alerts:
            hist[a.severity] = hist.get(a.severity, 0) + 1
        return hist

    @property
    def was_fully_thinned(self) -> bool:
        """True if the attacker acted but every alert was thinned away."""
        return (self.kind == "true_positive"
                and self.raw_cluster_size > 0
                and self.detected_size == 0)


# =====================================================================
# The generator
# =====================================================================

class AlertGenerator:
    """Four-layer realistic IDS alert-stream generator."""

    _DEFAULT_LOGIT = {
        "beta0": -1.5,
        "av": 4.0,
        "ac": 0.0,
        "auth": -1.0,
        "stealth": -3.0,
    }

    _DEFAULT_TIMING_PARAMS = {
        "poisson_rate": 5.0,
        "periodic_period": 0.4,
        "periodic_jitter": 0.1,
        "burst_span": 0.5,
    }

    def __init__(
        self,
        hosts: Sequence[str],
        exploits: Dict[str, Exploit],
        false_positive_rate: float = 0.30,
        *,
        fp_volume_scale: float = 1.0,
        fp_concentration: float = 0.65,
        fp_zipf_exponent: float = 1.0,
        fp_noisy_fraction: float = 0.3,
        hawkes_mu: float = 0.3,
        hawkes_alpha: float = 1.2,
        hawkes_beta: float = 1.5,
        default_timing_params: Optional[Dict[str, float]] = None,
        severity_weights: Optional[Dict[str, Sequence[float]]] = None,
        default_severity_weights: Sequence[float] = (0.1, 0.25, 0.3, 0.25, 0.1),
        enable_clustering: bool = True,
        enable_timing: bool = True,
        enable_thinning: bool = True,
        enable_fp_concentration: bool = True,
        logistic_coefficients: Optional[Dict[str, float]] = None,
        step_duration: float = 1.0,
        seed: Optional[int] = None,
    ):
        if len(hosts) == 0:
            raise ValueError("`hosts` must be non-empty.")
        if len(exploits) == 0:
            raise ValueError("`exploits` must be non-empty.")
        if not (0.0 <= fp_concentration <= 1.0):
            raise ValueError("`fp_concentration` must be in [0, 1].")
        if not (0.0 < fp_noisy_fraction <= 1.0):
            raise ValueError("`fp_noisy_fraction` must be in (0, 1].")
        if hawkes_alpha / hawkes_beta >= 1.0:
            raise ValueError(
                "Hawkes branching ratio alpha/beta must be < 1.")
        if step_duration <= 0:
            raise ValueError("`step_duration` must be > 0.")

        self.hosts: List[str] = list(hosts)
        self.H: int = len(self.hosts)
        self._host_to_idx = {h: i for i, h in enumerate(self.hosts)}
        self.exploits: Dict[str, Exploit] = dict(exploits)

        self.fp_volume_scale = float(fp_volume_scale)
        self.fp_concentration = float(fp_concentration)
        self.fp_zipf_exponent = float(fp_zipf_exponent)
        self.fp_noisy_fraction = float(fp_noisy_fraction)

        self.hawkes_mu = float(hawkes_mu)
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = float(hawkes_beta)

        self.timing_defaults = dict(self._DEFAULT_TIMING_PARAMS)
        if default_timing_params:
            unknown = set(default_timing_params) - set(self.timing_defaults)
            if unknown:
                raise ValueError(
                    f"Unknown default_timing_params keys: {sorted(unknown)}.")
            self.timing_defaults.update(default_timing_params)

        self.default_severity_weights = self._normalise(
            np.asarray(default_severity_weights, dtype=float),
            "default_severity_weights")
        self.severity_weights: Dict[str, np.ndarray] = {}
        if severity_weights:
            for sig, w in severity_weights.items():
                self.severity_weights[sig] = self._normalise(
                    np.asarray(w, dtype=float),
                    f"severity_weights['{sig}']")

        self.enable_clustering = bool(enable_clustering)
        self.enable_timing = bool(enable_timing)
        self.enable_thinning = bool(enable_thinning)
        self.enable_fp_concentration = bool(enable_fp_concentration)

        self.logit = dict(self._DEFAULT_LOGIT)
        if logistic_coefficients:
            unknown = set(logistic_coefficients) - set(self.logit)
            if unknown:
                raise ValueError(
                    f"Unknown logistic_coefficients keys: {sorted(unknown)}.")
            self.logit.update(logistic_coefficients)

        self.step_duration = float(step_duration)

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self._false_positive_rate = None
        self.set_false_positive_rate(false_positive_rate)

        self._hawkes_history: List[float] = []

    # ---- public configuration ---------------------------------------
    def set_false_positive_rate(self, rate: float) -> None:
        """Set FP rate and re-derive the noisy-host subset."""
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"false_positive_rate must be in [0, 1].")
        self._false_positive_rate = float(rate)

        self._n_noise_targets = max(1, int(np.floor(rate * self.H)))
        n_noisy = max(1, int(np.ceil(
            self.fp_noisy_fraction * self._n_noise_targets)))
        n_noisy = min(n_noisy, self._n_noise_targets)

        perm = self._rng.permutation(self.H)
        self._noise_target_idx = np.sort(perm[: self._n_noise_targets])
        self._noisy_idx = np.sort(self._noise_target_idx[:n_noisy])
        self._quiet_target_idx = np.sort(self._noise_target_idx[n_noisy:])

        ranks = np.arange(1, len(self._noisy_idx) + 1, dtype=float)
        zipf = ranks ** (-self.fp_zipf_exponent)
        self._zipf_weights = zipf / zipf.sum()

    @property
    def false_positive_rate(self) -> float:
        return self._false_positive_rate

    @property
    def branching_ratio(self) -> float:
        return self.hawkes_alpha / self.hawkes_beta

    def expected_fp_per_step(self) -> float:
        return self._false_positive_rate * self.fp_volume_scale * self.H

    def noisy_hosts(self) -> List[str]:
        """Current chronically-noisy host subset (for inspection)."""
        return [self.hosts[i] for i in self._noisy_idx]

    def reset_episode(self,
                      false_positive_rate: Optional[float] = None) -> None:
        """
        Call at the start of every episode. Clears Hawkes carry-over and
        re-draws the noisy-host subset (so noise identity is fixed within
        an episode, varies across episodes).
        """
        if false_positive_rate is not None:
            self.set_false_positive_rate(false_positive_rate)
        else:
            self.set_false_positive_rate(self._false_positive_rate)
        self._hawkes_history.clear()

    def reset_hawkes(self) -> None:
        self._hawkes_history.clear()

    # ---- main entry point -------------------------------------------
    def step(
        self,
        step_index: int,
        attacker_exploit: Optional[str] = None,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
    ) -> Tuple[Optional[AlertGroup], AlertGroup]:
        t0 = step_index * self.step_duration

        tp_group: Optional[AlertGroup] = None
        if attacker_exploit is not None:
            if attacker_exploit not in self.exploits:
                raise KeyError(
                    f"Unknown exploit '{attacker_exploit}'.")
            if src_ip is None or dst_ip is None:
                raise ValueError(
                    "src_ip and dst_ip required with attacker_exploit.")
            tp_group = self._generate_true_positive_group(
                self.exploits[attacker_exploit],
                step_index, t0, src_ip, dst_ip)

        fp_group = self._generate_false_positive_group(step_index, t0)
        return tp_group, fp_group

    # ---- Layers 1-3 : true-positive cluster -------------------------
    def _generate_true_positive_group(
        self, exploit: Exploit, step_index: int,
        t0: float, src_ip: str, dst_ip: str,
    ) -> AlertGroup:
        # Layer 1: cluster size
        if self.enable_clustering:
            raw_size = int(np.ceil(self._rng.lognormal(
                exploit.mu_cluster, exploit.sigma_cluster)))
            raw_size = max(raw_size, 1)
        else:
            raw_size = 1

        # Layer 3: detection probability + thinning
        # (drawn fresh per traversal -- not a per-node constant)
        p_detect = self._p_detect(exploit) if self.enable_thinning else 1.0
        detected_size = int(self._rng.binomial(raw_size, p_detect)) \
            if raw_size > 0 else 0

        # Layer 2: per-exploit timing dispatch
        if detected_size > 0:
            if self.enable_timing:
                times = self._timing_dispatch(exploit, t0, detected_size)
            else:
                times = np.sort(self._rng.uniform(
                    t0, t0 + self.step_duration, detected_size))
            self._hawkes_history.extend(times.tolist())
        else:
            times = np.array([])

        alerts: List[Alert] = []
        sig_ids, sig_probs = self._signature_dist(exploit)
        for i in range(detected_size):
            sig = self._rng.choice(sig_ids, p=sig_probs)
            sev = self._draw_severity(sig, exploit)
            alerts.append(Alert(
                time=float(times[i]),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=int(self._rng.integers(1024, 65535)),
                dst_port=self._dst_port_for(exploit),
                signature=sig,
                category=exploit.category,
                severity=int(sev),
                is_true_positive=True,
                exploit_name=exploit.name,
            ))

        return AlertGroup(
            alerts=alerts,
            step=step_index,
            raw_cluster_size=raw_size,
            detected_size=detected_size,
            kind="true_positive",
            exploit_name=exploit.name,
            timing_model=exploit.timing_model.value,
        )

    # ---- Layer 2 : per-exploit timing dispatch ----------------------
    def _timing_dispatch(self, exploit: Exploit, t0: float,
                         n: int) -> np.ndarray:
        """Produce n sorted timestamps via the exploit's timing model."""
        m = exploit.timing_model
        p = exploit.timing_params
        if m == TimingModel.HAWKES:
            return self._times_hawkes(t0, n, p)
        if m == TimingModel.POISSON:
            return self._times_poisson(t0, n, p)
        if m == TimingModel.PERIODIC:
            return self._times_periodic(t0, n, p)
        if m == TimingModel.BURST:
            return self._times_burst(t0, n, p)
        raise ValueError(f"Unhandled timing model: {m}")

    def _times_hawkes(self, t0: float, n: int,
                      p: Dict[str, float]) -> np.ndarray:
        """
        Self-exciting cascade seeded at t0. The exploit event acts as an
        immortal parent that lifts the intensity by alpha at t0; the
        process evolves by Ogata thinning until n events are produced.
        Per-exploit overrides: 'mu', 'alpha', 'beta'.
        """
        if n <= 0:
            return np.array([])

        mu = float(p.get("mu", self.hawkes_mu))
        alpha = float(p.get("alpha", self.hawkes_alpha))
        beta = float(p.get("beta", self.hawkes_beta))
        if alpha / beta >= 1.0:
            beta = alpha + 1e-3

        events: List[float] = []
        t = t0
        decay_horizon = 20.0 / beta
        hist = [ti for ti in self._hawkes_history
                if ti > t0 - decay_horizon]

        def intensity(at: float) -> float:
            lam = mu
            lam += alpha * np.exp(-beta * (at - t0))  # the exploit seed
            for ti in hist:
                if ti < at:
                    lam += alpha * np.exp(-beta * (at - ti))
            for ti in events:
                if ti < at:
                    lam += alpha * np.exp(-beta * (at - ti))
            return lam

        guard, max_guard = 0, 100 * n + 1000
        while len(events) < n and guard < max_guard:
            guard += 1
            lam_bar = intensity(t) + alpha
            if lam_bar <= 0:
                break
            t = t + self._rng.exponential(1.0 / lam_bar)
            if self._rng.uniform() * lam_bar <= intensity(t):
                events.append(t)

        if len(events) < n:
            last = events[-1] if events else t0
            shortfall = n - len(events)
            filler = last + np.cumsum(self._rng.exponential(
                1.0 / max(mu, 1e-3), shortfall))
            events.extend(filler.tolist())

        return np.sort(np.asarray(events[:n], dtype=float))

    def _times_poisson(self, t0: float, n: int,
                       p: Dict[str, float]) -> np.ndarray:
        """
        Homogeneous Poisson: exponential inter-arrivals at rate `rate`
        events/sec. Fano factor ~ 1. No clustering beyond chance.
        """
        rate = float(p.get("rate", self.timing_defaults["poisson_rate"]))
        rate = max(rate, 1e-6)
        gaps = self._rng.exponential(1.0 / rate, n)
        return t0 + np.cumsum(gaps)

    def _times_periodic(self, t0: float, n: int,
                        p: Dict[str, float]) -> np.ndarray:
        """
        Near-periodic: events at `period` apart, perturbed by uniform
        jitter in [-jitter, +jitter]. UNDER-dispersed (CV < 1) -- the
        signature of a brute-force or C2 beacon. Sorted in case large
        jitter swaps adjacent events.
        """
        period = float(p.get(
            "period", self.timing_defaults["periodic_period"]))
        jitter = float(p.get(
            "jitter", self.timing_defaults["periodic_jitter"]))
        period = max(period, 1e-6)
        base = t0 + period * np.arange(1, n + 1)
        noise = self._rng.uniform(-jitter, jitter, n)
        return np.sort(base + noise)

    def _times_burst(self, t0: float, n: int,
                     p: Dict[str, float]) -> np.ndarray:
        """
        One tight clump: n alerts uniformly within a short `span` after
        t0. The signature of a single-shot exploit.
        """
        span = float(p.get("span", self.timing_defaults["burst_span"]))
        span = max(span, 1e-6)
        return np.sort(t0 + self._rng.uniform(0.0, span, n))

    # ---- Layer 4 : false-positive mixture ---------------------------
    def _generate_false_positive_group(
        self, step_index: int, t0: float
    ) -> AlertGroup:
        mean_fp = self.expected_fp_per_step()
        n_fp = 0 if mean_fp <= 1e-9 else self._negbinom_by_mean(
            mean_fp, dispersion=2.0)

        alerts: List[Alert] = []
        for _ in range(n_fp):
            host = self.hosts[self._sample_fp_host()]
            sig = "POLICY Internal Host Policy Violation"
            sev = self._draw_severity(sig, exploit=None)
            jitter = self._rng.uniform(0.0, self.step_duration)
            alerts.append(Alert(
                time=float(t0 + jitter),
                src_ip=host,
                dst_ip=host,
                src_port=int(self._rng.integers(10000, 60000)),
                dst_port=53,
                signature=sig,
                category="Policy Violation",
                severity=int(sev),
                is_true_positive=False,
                exploit_name=None,
            ))
        alerts.sort(key=lambda a: a.time)

        return AlertGroup(
            alerts=alerts,
            step=step_index,
            raw_cluster_size=n_fp,
            detected_size=n_fp,
            kind="false_positive",
            exploit_name=None,
            timing_model=None,
        )

    def _sample_fp_host(self) -> int:
        if not self.enable_fp_concentration:
            return int(self._rng.integers(0, self.H))
        if self._rng.uniform() < self.fp_concentration \
                and len(self._noisy_idx) > 0:
            local = self._rng.choice(
                len(self._noisy_idx), p=self._zipf_weights)
            return int(self._noisy_idx[local])
        if len(self._quiet_target_idx) > 0:
            return int(self._rng.choice(self._quiet_target_idx))
        return int(self._rng.choice(self._noisy_idx))

    # ---- Layer 3 : logistic detection -------------------------------
    def _p_detect(self, exploit: Exploit) -> float:
        x = (self.logit["beta0"]
             + self.logit["av"] * exploit.av
             + self.logit["ac"] * exploit.ac
             + self.logit["auth"] * exploit.auth
             + self.logit["stealth"] * exploit.stealth)
        return float(1.0 / (1.0 + np.exp(-x)))

    def detection_probability(self, exploit_name: str) -> float:
        if exploit_name not in self.exploits:
            raise KeyError(f"Unknown exploit '{exploit_name}'.")
        return self._p_detect(self.exploits[exploit_name])

    # ---- helpers ----------------------------------------------------
    def _signature_dist(self, exploit: Exploit
                        ) -> Tuple[List[str], np.ndarray]:
        if exploit.signature_weights:
            ids = list(exploit.signature_weights.keys())
            probs = np.asarray([exploit.signature_weights[k]
                                for k in ids], dtype=float)
        else:
            ids = [f"{exploit.name}:generic"]
            probs = np.array([1.0])
        return ids, probs

    def _draw_severity(self, signature: str,
                       exploit: Optional[Exploit]) -> int:
        if signature in self.severity_weights:
            weights = self.severity_weights[signature]
        elif exploit is not None and exploit.severity_weights is not None:
            weights = np.asarray(exploit.severity_weights, dtype=float)
        else:
            weights = self.default_severity_weights
        return int(self._rng.choice(
            np.arange(1, len(weights) + 1), p=weights))

    def _dst_port_for(self, exploit: Exploit) -> int:
        if exploit.av >= 0.5:
            return int(self._rng.choice([80, 443, 445, 22, 3389, 8080]))
        return int(self._rng.integers(1024, 65535))

    def _negbinom_by_mean(self, mean: float, dispersion: float) -> int:
        if dispersion <= 1.0:
            return int(self._rng.poisson(mean))
        p = 1.0 / dispersion
        r = mean * p / (1.0 - p)
        if r <= 0:
            return 0
        return int(self._rng.negative_binomial(r, p))

    @staticmethod
    def _normalise(weights: np.ndarray, name: str) -> np.ndarray:
        if np.any(weights < 0):
            raise ValueError(f"{name}: weights must be non-negative.")
        total = weights.sum()
        if total <= 0:
            raise ValueError(f"{name}: weights sum to <= 0.")
        return weights / total

    def run_episode(
        self,
        attacker_plan: Sequence[Optional[Tuple[str, str, str]]],
        false_positive_rate: Optional[float] = None,
    ) -> List[Tuple[Optional[AlertGroup], AlertGroup]]:
        self.reset_episode(false_positive_rate)
        out = []
        for i, action in enumerate(attacker_plan):
            if action is None:
                out.append(self.step(i))
            else:
                exploit_name, src_ip, dst_ip = action
                out.append(self.step(i, exploit_name, src_ip, dst_ip))
        return out


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    import numpy as _np

    def _gini(x):
        x = _np.sort(_np.asarray(x, dtype=float))
        n = len(x)
        if x.sum() == 0:
            return 0.0
        return (2 * _np.sum(_np.arange(1, n + 1) * x)
                - (n + 1) * x.sum()) / (n * x.sum())

    def _cv(times):
        t = _np.sort(_np.asarray(times))
        if len(t) < 3:
            return float("nan")
        iat = _np.diff(t)
        return iat.std() / iat.mean() if iat.mean() > 0 else float("nan")

    print("=" * 72)
    print("AlertGenerator self-test  --  four layers + per-exploit timing")
    print("=" * 72)

    hosts = [f"10.0.0.{i}" for i in range(1, 51)]

    exploits = {
        "scan": Exploit(
            "scan", av=1.0, ac=0.3, auth=0, stealth=0.0,
            mu_cluster=4.0, sigma_cluster=0.8,
            timing_model=TimingModel.HAWKES,
            category="Potential Network Scan",
            signature_weights={"ET-SCAN-portscan": 0.7,
                               "ET-SCAN-nmap": 0.3}),
        "rce": Exploit(
            "rce", av=1.0, ac=0.5, auth=0, stealth=0.2,
            mu_cluster=2.8, sigma_cluster=0.6,
            timing_model=TimingModel.BURST,
            timing_params={"span": 0.4},
            category="Network Exploitation Attempt",
            signature_weights={"ET-EXPLOIT-rce": 0.6,
                               "ET-WEB-attack": 0.4}),
        "brute": Exploit(
            "brute", av=1.0, ac=0.2, auth=0, stealth=0.1,
            mu_cluster=3.0, sigma_cluster=0.4,
            timing_model=TimingModel.PERIODIC,
            timing_params={"period": 0.5, "jitter": 0.05},
            category="Brute Force Attempt"),
        "cred_reuse": Exploit(
            "cred_reuse", av=0.0, ac=0.2, auth=1, stealth=1.0,
            mu_cluster=1.5, sigma_cluster=0.5,
            timing_model=TimingModel.POISSON,
            timing_params={"rate": 3.0},
            category="Local Exploitation Attempt"),
    }

    print("\n[0] false_positive_rate wiring")
    print(f"    {'fpr':>5} {'noise hosts':>12} {'noisy subset':>13} "
          f"{'E[FP/step]':>11}")
    for fpr in (0.0, 0.10, 0.30, 0.60, 0.90):
        g = AlertGenerator(hosts, exploits, false_positive_rate=fpr, seed=1)
        print(f"    {fpr:>5.2f} {g._n_noise_targets:>12d} "
              f"{len(g._noisy_idx):>13d} {g.expected_fp_per_step():>11.1f}")

    print("\n[1] Layer 3: logistic detection probability per exploit")
    g = AlertGenerator(hosts, exploits, false_positive_rate=0.30, seed=42)
    for name in exploits:
        print(f"    {name:12s}  P_detect = "
              f"{g.detection_probability(name):.3f}"
              f"   timing = {exploits[name].timing_model.value}")
    print("    (drawn fresh per traversal -- NOT a per-node constant)")

    print("\n[2] Layer 1: clustering ablation")
    g_off = AlertGenerator(hosts, exploits, false_positive_rate=0.0,
                           enable_clustering=False,
                           enable_thinning=False, seed=1)
    sizes_off = [g_off.step(i, "scan", "10.0.0.1", "10.0.0.2")[0].raw_cluster_size
                 for i in range(30)]
    g_on = AlertGenerator(hosts, exploits, false_positive_rate=0.0, seed=1)
    sizes_on = [g_on.step(i, "scan", "10.0.0.1", "10.0.0.2")[0].raw_cluster_size
                for i in range(30)]
    print(f"    clustering OFF: all raw sizes == 1 ? "
          f"{set(sizes_off) == {1}}")
    print(f"    clustering ON : raw sizes vary, range "
          f"[{min(sizes_on)}, {max(sizes_on)}]")

    print("\n[3] Layer 2: each timing model has a distinct signature")
    print("    (Poisson CV~1 ; Hawkes CV>1 bursty ; Periodic CV<1 regular ;")
    print("     Burst CV~0.6 -- uniform-in-short-window)")
    for name in ("scan", "rce", "brute", "cred_reuse"):
        g_t = AlertGenerator(hosts, exploits, false_positive_rate=0.0,
                             enable_thinning=False, seed=11)
        cvs, spans = [], []
        for i in range(150):
            tp, _ = g_t.step(i, name, "10.0.0.1", "10.0.0.2")
            if tp and tp.detected_size > 5:
                tms = sorted(a.time for a in tp.alerts)
                cvs.append(_cv(tms))
                spans.append(tms[-1] - tms[0])
        model = exploits[name].timing_model.value
        print(f"    {name:12s} [{model:8s}]  mean IAT CV = "
              f"{_np.mean(cvs):.3f}   mean cluster span = "
              f"{_np.mean(spans):6.2f}s")

    print("\n[4] Layer 4: false-positive concentration ablation (Gini)")
    for flag in (True, False):
        g = AlertGenerator(hosts, exploits, false_positive_rate=0.30,
                           enable_fp_concentration=flag, seed=7)
        fph = _np.zeros(g.H)
        total = 0
        for i in range(2000):
            _, fp = g.step(i)
            for a in fp.alerts:
                fph[g._host_to_idx[a.src_ip]] += 1
            total += len(fp.alerts)
        print(f"    concentration {str(flag):5s} -> Gini = "
              f"{_gini(fph):.3f}   (mean FP/step = {total/2000:.1f}, "
              f"target = {g.expected_fp_per_step():.1f})")
    print("    (ON ~0.82 = operational SOC regime; OFF ~0.02 = uniform)")

    print("\n[5] noisy-host subset fixed within episode, varies across")
    g = AlertGenerator(hosts, exploits, false_positive_rate=0.30, seed=3)
    subsets = []
    for ep in range(4):
        g.reset_episode()
        subsets.append(tuple(g.noisy_hosts()))
    for ep, s in enumerate(subsets):
        print(f"    episode {ep}: noisy hosts = {list(s)}")
    print(f"    all four episodes identical? "
          f"{len(set(subsets)) == 1}  (should be False)")

    print("\n[6] end-to-end 6-step episode")
    g = AlertGenerator(hosts, exploits, false_positive_rate=0.30, seed=42)
    plan = [
        ("scan",       "10.0.0.5",  "10.0.0.12"),
        ("rce",        "10.0.0.5",  "10.0.0.12"),
        None,
        ("cred_reuse", "10.0.0.12", "10.0.0.20"),
        ("brute",      "10.0.0.20", "10.0.0.30"),
        ("rce",        "10.0.0.20", "10.0.0.30"),
    ]
    for i, (tp, fp) in enumerate(g.run_episode(plan)):
        if tp is None:
            desc = "idle"
        else:
            desc = (f"{tp.exploit_name:11s} [{tp.timing_model:8s}] "
                    f"raw={tp.raw_cluster_size:>3d} "
                    f"detected={tp.detected_size:>3d}")
            if tp.was_fully_thinned:
                desc += "  <- acted but IDS saw NOTHING"
        print(f"    step {i}: TP[{desc:<62s}] FP[{len(fp.alerts):>2d}]")

    print("\n" + "=" * 72)
    print("Self-test complete.")
    print("=" * 72)
