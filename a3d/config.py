# FILE: a3d/config.py
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class AegisConfig:
    """Global configuration for Aegis decoders and graph building."""

    distance: int = 5
    rounds: int = 6

    # greedy | osd | mwpm | mwpm2 | mwpm_corr | uf | unionfind | uf_e
    decoder_type: str = "mwpm"

    p_data: float = 0.02
    p_meas: float = 0.03
    p_leak: float = 0.0
    time_weight_scale: float = 1.0

    uf_confidence_threshold: float = 3.0
    avg_cost_threshold: float = 5.0

    # none | transformer | transformer_sota | bp
    reweighter_type: str = "none"
    reweighter_weights: str = ""

    correlation_params: str = ""
    log_events: bool = False
    log_path: str = "logs/decoding_events.csv"

    run_certificate: bool = False
    certificate_mode: str = "osd"
    profile: bool = False
    profile_path: str = "logs/profile.csv"

    # Kept for configuration-file compatibility. MWPM now always uses
    # PyMatching sparse blossom; there is no NetworkX production fallback.
    prefer_pymatching: bool = True

    def save_to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "AegisConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def validate(self) -> None:
        if not (isinstance(self.distance, int) and self.distance >= 3):
            raise ValueError("distance must be an integer >= 3")
        if not (isinstance(self.rounds, int) and self.rounds >= 2):
            raise ValueError("rounds must be an integer >= 2")

        valid_decoders = {
            "greedy",
            "osd",
            "mwpm",
            "mwpm2",
            "pipelined_mwpm",
            "pmwpm",
            "mwpm_corr",
            "mwpmc",
            "mwpm3",
            "uf",
            "unionfind",
            "uf_e",
        }
        if self.decoder_type not in valid_decoders:
            raise ValueError(
                "decoder_type must be one of " + ", ".join(sorted(valid_decoders))
            )

        for name in (
            "p_data",
            "p_meas",
            "p_leak",
            "time_weight_scale",
            "uf_confidence_threshold",
            "avg_cost_threshold",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1e6):
                raise ValueError(f"{name} out of bounds: {value}")
            if name in {"p_data", "p_meas", "p_leak"} and value >= 1.0:
                raise ValueError(f"{name} must be a probability below one")
            if name == "time_weight_scale" and value <= 0.0:
                raise ValueError("time_weight_scale must be positive")

        valid_reweighters = {"none", "transformer", "transformer_sota", "bp"}
        if self.reweighter_type not in valid_reweighters:
            raise ValueError(
                "reweighter_type must be one of "
                + ", ".join(sorted(valid_reweighters))
            )
