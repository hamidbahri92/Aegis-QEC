from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .research import wilson_interval


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _fit_logistic_coefficients(
    x: np.ndarray,
    failures: np.ndarray,
    shots: np.ndarray,
    *,
    max_iterations: int = 80,
) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=float)

    for _ in range(max_iterations):
        logits = design @ beta
        probability = np.clip(_sigmoid(logits), 1.0e-12, 1.0 - 1.0e-12)
        gradient = design.T @ (failures - shots * probability)
        weight = shots * probability * (1.0 - probability)
        information = design.T @ (design * weight[:, None])
        information += np.eye(2) * 1.0e-10
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
        beta += step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break

    probability = np.clip(_sigmoid(design @ beta), 1.0e-12, 1.0 - 1.0e-12)
    log_likelihood = float(
        np.sum(
            failures * np.log(probability)
            + (shots - failures) * np.log1p(-probability)
        )
    )
    return beta, log_likelihood


def _scaling_x(
    physical_error_rate: np.ndarray,
    distance: np.ndarray,
    critical_probability: float,
    nu: float,
) -> np.ndarray:
    return (
        physical_error_rate - float(critical_probability)
    ) * np.power(distance, 1.0 / float(nu))


def _extract_campaign_points(
    campaign: dict[str, Any],
    *,
    decoder: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in campaign.get("rows", []):
        if str(row.get("decoder")) != str(decoder):
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("distance") is None:
            continue
        if metadata.get("physical_error_rate") is None:
            continue

        shots = int(row.get("accepted_shots", row.get("shots", 0)))
        failures = int(row.get("errors", 0))
        if shots <= 0 or failures < 0 or failures > shots:
            continue
        points.append(
            {
                "distance": int(metadata["distance"]),
                "physical_error_rate": float(metadata["physical_error_rate"]),
                "shots": shots,
                "failures": failures,
                "logical_error_rate": float(failures / shots),
            }
        )
    return points


def _validate_scaling_points(points: Sequence[dict[str, Any]]) -> None:
    distances = sorted({int(point["distance"]) for point in points})
    probabilities = sorted(
        {float(point["physical_error_rate"]) for point in points}
    )
    if len(distances) < 3:
        raise ValueError(
            "Finite-size scaling requires at least three code distances."
        )
    if len(probabilities) < 4:
        raise ValueError(
            "Finite-size scaling requires at least four physical-error points."
        )
    total_shots = sum(int(point["shots"]) for point in points)
    total_failures = sum(int(point["failures"]) for point in points)
    if total_failures <= 0 or total_failures >= total_shots:
        raise ValueError(
            "Scaling data must contain both logical successes and failures."
        )


def fit_surface_code_scaling(
    campaign: dict[str, Any],
    *,
    decoder: str,
    nu_min: float = 0.5,
    nu_max: float = 3.0,
    pc_grid_points: int = 61,
    nu_grid_points: int = 31,
) -> dict[str, Any]:
    """Fit a first-order finite-size scaling model to campaign statistics.

    The fitted model is

        logit(P_L) = beta_0 + beta_1 * (p - p_c) * d ** (1 / nu)

    using the binomial likelihood of the aggregated logical-failure counts.
    The confidence interval for p_c is a profile-likelihood interval based on
    a chi-square-one 95 percent cutoff.
    """

    points = _extract_campaign_points(campaign, decoder=decoder)
    _validate_scaling_points(points)

    p = np.asarray(
        [point["physical_error_rate"] for point in points],
        dtype=float,
    )
    d = np.asarray([point["distance"] for point in points], dtype=float)
    n = np.asarray([point["shots"] for point in points], dtype=float)
    k = np.asarray([point["failures"] for point in points], dtype=float)

    p_min = float(np.min(p))
    p_max = float(np.max(p))
    if not p_max > p_min:
        raise ValueError("physical-error grid has zero width")
    if not 0.0 < nu_min < nu_max:
        raise ValueError("require 0 < nu_min < nu_max")
    if pc_grid_points < 21 or nu_grid_points < 11:
        raise ValueError("scaling grids are too small for a useful fit")

    pc_grid = np.linspace(p_min, p_max, int(pc_grid_points))
    nu_grid = np.linspace(float(nu_min), float(nu_max), int(nu_grid_points))

    profile_log_likelihood = np.full(pc_grid.shape, -np.inf, dtype=float)
    profile_nu = np.full(pc_grid.shape, np.nan, dtype=float)
    profile_beta = np.full((len(pc_grid), 2), np.nan, dtype=float)

    best_log_likelihood = -np.inf
    best_pc = math.nan
    best_nu = math.nan
    best_beta = np.zeros(2, dtype=float)

    for pc_index, critical_probability in enumerate(pc_grid):
        for nu in nu_grid:
            x = _scaling_x(p, d, critical_probability, float(nu))
            beta, log_likelihood = _fit_logistic_coefficients(x, k, n)
            if log_likelihood > profile_log_likelihood[pc_index]:
                profile_log_likelihood[pc_index] = log_likelihood
                profile_nu[pc_index] = float(nu)
                profile_beta[pc_index] = beta
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_pc = float(critical_probability)
                best_nu = float(nu)
                best_beta = beta

    pc_step = float(pc_grid[1] - pc_grid[0])
    nu_step = float(nu_grid[1] - nu_grid[0])
    refined_pc = np.linspace(
        max(p_min, best_pc - 2.0 * pc_step),
        min(p_max, best_pc + 2.0 * pc_step),
        41,
    )
    refined_nu = np.linspace(
        max(nu_min, best_nu - 2.0 * nu_step),
        min(nu_max, best_nu + 2.0 * nu_step),
        31,
    )
    # Profile the likelihood at every refined critical probability, not
    # merely the winner. Otherwise the 95% interval still uses the coarse
    # grid while the reported point estimate comes from the refined grid.
    refined_profile_ll = np.full(refined_pc.shape, -np.inf, dtype=float)
    for index, critical_probability in enumerate(refined_pc):
        for nu in refined_nu:
            x = _scaling_x(p, d, float(critical_probability), float(nu))
            beta, log_likelihood = _fit_logistic_coefficients(x, k, n)
            refined_profile_ll[index] = max(
                refined_profile_ll[index], log_likelihood
            )
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_pc = float(critical_probability)
                best_nu = float(nu)
                best_beta = beta

    cutoff = best_log_likelihood - 0.5 * 3.841458820694124
    profiled_pc = np.concatenate((pc_grid, refined_pc))
    profiled_ll = np.concatenate((profile_log_likelihood, refined_profile_ll))
    supported_pc = profiled_pc[profiled_ll >= cutoff]
    if len(supported_pc):
        ci_low = min(float(np.min(supported_pc)), best_pc)
        ci_high = max(float(np.max(supported_pc)), best_pc)
    else:
        ci_low = best_pc
        ci_high = best_pc

    # A profile-likelihood interval is mathematically closed. Round its
    # floating-point representation outward by one ULP so an endpoint such as
    # decimal 0.01 is not spuriously reported as 0.009999999999999998.
    reported_ci_low = math.nextafter(ci_low, -math.inf)
    reported_ci_high = math.nextafter(ci_high, math.inf)

    null_x = p
    _, null_log_likelihood = _fit_logistic_coefficients(null_x, k, n)
    scaling_aic = 2.0 * 4.0 - 2.0 * best_log_likelihood
    null_aic = 2.0 * 2.0 - 2.0 * null_log_likelihood

    warnings: list[str] = []
    if best_pc <= p_min + pc_step or best_pc >= p_max - pc_step:
        warnings.append(
            "Best-fit critical probability is near the sampled grid boundary; "
            "expand the physical-error grid before interpreting it as a threshold."
        )
    if ci_low <= p_min or ci_high >= p_max:
        warnings.append(
            "The profile-likelihood interval reaches the sampled grid boundary; "
            "the campaign does not tightly identify the critical probability."
        )
    if best_nu <= nu_min + nu_step or best_nu >= nu_max - nu_step:
        warnings.append(
            "Best-fit nu is near the configured search boundary; widen the nu range."
        )
    if float(best_beta[1]) <= 0.0:
        warnings.append(
            "The fitted logical-error slope is non-positive, which is inconsistent "
            "with the expected monotonic scaling trend."
        )
    if scaling_aic >= null_aic:
        warnings.append(
            "The finite-size scaling model is not preferred over a simple "
            "distance-independent logistic trend by AIC."
        )

    return {
        "schema_version": 1,
        "analysis_type": "first_order_finite_size_scaling",
        "decoder": str(decoder),
        "model": (
            "logit(P_L) = beta_0 + beta_1 * "
            "(p - p_c) * distance ** (1 / nu)"
        ),
        "critical_probability": best_pc,
        "critical_probability_ci95_profile": [
            reported_ci_low,
            reported_ci_high,
        ],
        "nu": best_nu,
        "beta_0": float(best_beta[0]),
        "beta_1": float(best_beta[1]),
        "log_likelihood": float(best_log_likelihood),
        "null_log_likelihood": float(null_log_likelihood),
        "aic": float(scaling_aic),
        "null_aic": float(null_aic),
        "delta_aic_null_minus_scaling": float(null_aic - scaling_aic),
        "sampled_probability_range": [p_min, p_max],
        "distances": sorted({int(point["distance"]) for point in points}),
        "physical_error_rates": sorted(
            {float(point["physical_error_rate"]) for point in points}
        ),
        "total_shots": int(np.sum(n)),
        "total_logical_failures": int(np.sum(k)),
        "points_used": len(points),
        "warnings": warnings,
        "interpretation": (
            "This is a first-order finite-size scaling fit to simulated campaign "
            "statistics. The profile interval quantifies uncertainty inside the "
            "chosen model and sampled grid. It is not a hardware threshold unless "
            "the input data themselves come from a validated hardware experiment."
        ),
    }


def load_campaign_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("campaign JSON must contain an object")
    return value


def write_scaling_artifacts(
    campaign: dict[str, Any],
    analysis: dict[str, Any],
    *,
    json_path: str | None = "research_out/scaling.json",
    plot_path: str | None = "research_out/scaling.png",
) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if not plot_path:
        return

    from .plotting import new_agg_figure

    points = _extract_campaign_points(
        campaign,
        decoder=str(analysis["decoder"]),
    )
    path = Path(plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = new_agg_figure(figsize=(7.5, 5.0))

    critical_probability = float(analysis["critical_probability"])
    nu = float(analysis["nu"])
    beta = np.asarray(
        [float(analysis["beta_0"]), float(analysis["beta_1"])],
        dtype=float,
    )
    p_min, p_max = [
        float(value) for value in analysis["sampled_probability_range"]
    ]
    smooth_p = np.linspace(p_min, p_max, 200)

    for distance in analysis["distances"]:
        selected = sorted(
            (
                point
                for point in points
                if int(point["distance"]) == int(distance)
            ),
            key=lambda point: float(point["physical_error_rate"]),
        )
        x_values = [
            float(point["physical_error_rate"]) for point in selected
        ]
        y_values = [float(point["logical_error_rate"]) for point in selected]
        intervals = [
            wilson_interval(int(point["failures"]), int(point["shots"]))
            for point in selected
        ]
        lower = [
            y - interval[0]
            for y, interval in zip(y_values, intervals, strict=True)
        ]
        upper = [
            interval[1] - y
            for y, interval in zip(y_values, intervals, strict=True)
        ]
        ax.errorbar(
            x_values,
            y_values,
            yerr=[lower, upper],
            fmt="o",
            capsize=3,
            label=f"d={distance} data",
        )

        smooth_d = np.full_like(smooth_p, float(distance))
        smooth_x = _scaling_x(
            smooth_p,
            smooth_d,
            critical_probability,
            nu,
        )
        smooth_probability = _sigmoid(beta[0] + beta[1] * smooth_x)
        ax.plot(
            smooth_p,
            smooth_probability,
            label=f"d={distance} fit",
        )

    ax.axvline(
        critical_probability,
        linestyle="--",
        label=f"p_c={critical_probability:.6g}",
    )
    ax.set_xlabel("Physical error probability p")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Aegis QEC finite-size scaling analysis")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.clear()
