"""
ICS2 Week 6 - Cybersecurity Architecture Module

STRIDE threat model for sensor-driven corrosion monitoring in
consumer electronics manufacturing and quality control.

Context: If this galvanic corrosion model is deployed as part of
a manufacturing quality prediction system or warranty analytics
pipeline, the following threats apply.

Threat surfaces:
1. Sensor data from accelerated life testing (ALT) stations
2. Model parameters stored in calibration databases
3. Prediction outputs feeding warranty cost models
4. Audit trail for regulatory compliance (EU Product Liability Directive)
"""

import hashlib
import json
import time
import numpy as np
from datetime import datetime


# ==============================================================================
# STRIDE Threat Model
# ==============================================================================

STRIDE_THREATS = {
    "Spoofing": {
        "threat": "Attacker injects falsified ALT sensor data to make defective "
                  "connector batches appear compliant.",
        "impact": "Defective products ship, warranty claims spike, brand damage.",
        "mitigation": "Cryptographic signing of sensor readings at source. "
                      "Hardware security module (HSM) in ALT stations.",
        "residual_risk": "Compromised HSM firmware. Mitigated by firmware attestation.",
    },
    "Tampering": {
        "threat": "Calibrated j0 parameters are modified in the database to "
                  "underpredict corrosion rates.",
        "impact": "Overly optimistic lifetime predictions, insufficient plating thickness.",
        "mitigation": "Hash-chain integrity on parameter database. Write-once audit log. "
                      "Dual-approval for parameter updates.",
        "residual_risk": "Insider threat with dual approvals. Mitigated by anomaly detection.",
    },
    "Repudiation": {
        "threat": "Engineer denies modifying model parameters that led to "
                  "incorrect warranty cost projection.",
        "impact": "Inability to trace root cause of prediction failure.",
        "mitigation": "SHA-256 hash-chain audit log with timestamps and user identity. "
                      "Immutable log storage.",
        "residual_risk": "Shared credentials. Mitigated by MFA and individual accounts.",
    },
    "Information_Disclosure": {
        "threat": "Competitor extracts proprietary plating stack parameters "
                  "(Au/Ni/Cu thickness, j0 calibration) from API.",
        "impact": "Loss of competitive advantage in connector reliability.",
        "mitigation": "API access control with role-based permissions. "
                      "Parameter obfuscation in external-facing outputs.",
        "residual_risk": "Reverse engineering from public warranty data.",
    },
    "Denial_of_Service": {
        "threat": "Flood of malformed sensor data overwhelms the prediction pipeline "
                  "during production ramp.",
        "impact": "Delayed quality decisions, production line stoppage.",
        "mitigation": "Input validation and rate limiting. Redundant prediction instances. "
                      "Circuit breaker pattern on data ingestion.",
        "residual_risk": "Sustained DDoS exceeding rate limits.",
    },
    "Elevation_of_Privilege": {
        "threat": "QC operator gains access to modify inverse estimation priors, "
                  "biasing the Bayesian calibration.",
        "impact": "Systematically biased model outputs.",
        "mitigation": "Principle of least privilege. Separate roles for data collection, "
                      "model calibration, and production prediction.",
        "residual_risk": "Privilege escalation via unpatched dependencies.",
    },
}


# ==============================================================================
# Audit Logging with SHA-256 Hash Chain
# ==============================================================================

class AuditLogger:
    """
    Immutable audit log with SHA-256 hash chain for parameter changes
    and prediction events.
    """

    def __init__(self):
        self.chain = []
        self.genesis_hash = hashlib.sha256(b"ICS2_WEEK6_GENESIS").hexdigest()

    def _get_previous_hash(self):
        if len(self.chain) == 0:
            return self.genesis_hash
        return self.chain[-1]["hash"]

    def log_event(self, event_type, user, data, metadata=None):
        """
        Add an event to the audit chain.

        Parameters
        ----------
        event_type : str
            Type of event (parameter_update, prediction, sensor_reading, etc.)
        user : str
            User or system identifier.
        data : dict
            Event payload.
        metadata : dict, optional
            Additional context.
        """
        entry = {
            "index": len(self.chain),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "user": user,
            "data": data,
            "metadata": metadata or {},
            "previous_hash": self._get_previous_hash(),
        }

        # Compute hash of this entry
        entry_bytes = json.dumps(entry, sort_keys=True, default=str).encode()
        entry["hash"] = hashlib.sha256(entry_bytes).hexdigest()

        self.chain.append(entry)
        return entry

    def verify_chain(self):
        """
        Verify integrity of the entire audit chain.

        Returns
        -------
        valid : bool
            True if chain is intact.
        broken_at : int or None
            Index where chain breaks, if any.
        """
        for i, entry in enumerate(self.chain):
            # Verify previous hash linkage
            if i == 0:
                expected_prev = self.genesis_hash
            else:
                expected_prev = self.chain[i - 1]["hash"]

            if entry["previous_hash"] != expected_prev:
                return False, i

            # Verify self-hash
            stored_hash = entry["hash"]
            entry_copy = dict(entry)
            del entry_copy["hash"]
            entry_bytes = json.dumps(entry_copy, sort_keys=True, default=str).encode()
            computed_hash = hashlib.sha256(entry_bytes).hexdigest()

            if computed_hash != stored_hash:
                return False, i

        return True, None

    def get_chain_summary(self):
        """Return summary of audit chain."""
        return {
            "total_entries": len(self.chain),
            "first_entry": self.chain[0]["timestamp"] if self.chain else None,
            "last_entry": self.chain[-1]["timestamp"] if self.chain else None,
            "chain_valid": self.verify_chain()[0],
        }


# ==============================================================================
# Sensor Data Validation
# ==============================================================================

class SensorValidator:
    """
    Validates incoming sensor data against physical constraints
    and statistical anomaly detection.
    """

    def __init__(self, spoofing_threshold=3.0, max_drift_rate=0.1):
        self.spoofing_threshold = spoofing_threshold
        self.max_drift_rate = max_drift_rate
        self.history = {}

    def validate_reading(self, sensor_id, value, expected_range,
                          physical_unit="A/m^2"):
        """
        Validate a sensor reading.

        Returns
        -------
        result : dict
            {'valid': bool, 'flags': list, 'value': float}
        """
        flags = []
        valid = True

        # Range check
        if value < expected_range[0] or value > expected_range[1]:
            flags.append(f"OUT_OF_RANGE: {value} {physical_unit} "
                         f"outside [{expected_range[0]}, {expected_range[1]}]")
            valid = False

        # Statistical anomaly check against history
        if sensor_id in self.history and len(self.history[sensor_id]) >= 5:
            hist = np.array(self.history[sensor_id][-50:])
            mu = np.mean(hist)
            sigma = np.std(hist)
            if sigma > 0:
                z_score = abs(value - mu) / sigma
                if z_score > self.spoofing_threshold:
                    flags.append(f"STATISTICAL_ANOMALY: z-score={z_score:.2f} "
                                 f"exceeds threshold={self.spoofing_threshold}")
                    valid = False

        # Drift rate check
        if sensor_id in self.history and len(self.history[sensor_id]) >= 2:
            prev = self.history[sensor_id][-1]
            drift = abs(value - prev) / max(abs(prev), 1e-20)
            if drift > self.max_drift_rate:
                flags.append(f"EXCESSIVE_DRIFT: {drift:.4f} exceeds "
                             f"max rate {self.max_drift_rate}")
                # Drift is flagged but not necessarily invalid
                # (could be legitimate rapid change)

        # Update history
        if sensor_id not in self.history:
            self.history[sensor_id] = []
        self.history[sensor_id].append(value)

        return {"valid": valid, "flags": flags, "value": value}

    def check_physical_consistency(self, currents_dict):
        """
        Check that total anodic current approximately equals total cathodic current
        (conservation of charge).

        Parameters
        ----------
        currents_dict : dict
            {metal_key: current_A}

        Returns
        -------
        consistent : bool
        imbalance_fraction : float
        """
        total_anodic = sum(v for v in currents_dict.values() if v > 0)
        total_cathodic = abs(sum(v for v in currents_dict.values() if v < 0))

        if total_anodic + total_cathodic < 1e-20:
            return True, 0.0

        imbalance = abs(total_anodic - total_cathodic) / max(total_anodic, total_cathodic)
        consistent = imbalance < 0.1  # 10% tolerance

        return consistent, imbalance


# ==============================================================================
# Data Poisoning Detection for ML/Inverse Models
# ==============================================================================

def detect_data_poisoning(obs_data, n_leave_out=5, threshold_factor=3.0):
    """
    Simple leave-one-out residual analysis to detect potentially
    poisoned observations in the calibration dataset.

    If removing a single observation dramatically changes the inverse
    estimation result, that observation is suspicious.

    Parameters
    ----------
    obs_data : list of dict
        Observation dataset.
    n_leave_out : int
        Number of random leave-one-out tests.
    threshold_factor : float
        Flag observations whose removal changes I_corr by more than
        threshold_factor * median_change.

    Returns
    -------
    suspicious_indices : list of int
        Indices of potentially poisoned observations.
    influence_scores : ndarray
        Influence score for each tested observation.
    """
    from .galvanic_coupling import solve_mixed_potential

    if len(obs_data) < 3:
        return [], np.array([])

    # Baseline: solve with all data
    baseline_I = np.mean([abs(obs["I_measured"]) for obs in obs_data])

    # Leave-one-out
    influence_scores = np.zeros(min(n_leave_out, len(obs_data)))
    test_indices = np.random.choice(len(obs_data), size=len(influence_scores),
                                     replace=False)

    for k, idx in enumerate(test_indices):
        reduced_data = [obs for j, obs in enumerate(obs_data) if j != idx]
        reduced_I = np.mean([abs(obs["I_measured"]) for obs in reduced_data])
        influence_scores[k] = abs(reduced_I - baseline_I) / max(baseline_I, 1e-20)

    median_influence = np.median(influence_scores)
    suspicious = []
    for k, idx in enumerate(test_indices):
        if influence_scores[k] > threshold_factor * max(median_influence, 1e-10):
            suspicious.append(int(idx))

    return suspicious, influence_scores
