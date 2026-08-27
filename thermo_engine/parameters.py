"""Explicit parameter direction transformations."""

from __future__ import annotations

import numpy as np

from schemas.domain import ComponentIdentity, ParameterSet


def reverse_binary_parameter_direction(
    parameters: dict[str, float], directional_pairs: list[tuple[str, str]]
) -> dict[str, float]:
    """Reverse named directional pairs without guessing a model's parameter form."""
    reversed_parameters = dict(parameters)
    for forward, reverse in directional_pairs:
        if forward not in parameters or reverse not in parameters:
            raise ValueError(f"Directional pair {forward}/{reverse} is incomplete")
        reversed_parameters[forward] = parameters[reverse]
        reversed_parameters[reverse] = parameters[forward]
    return reversed_parameters


def has_chemsep_kij(components: list[ComponentIdentity]) -> bool:
    """Return True when every binary pair has a reviewed ChemSep PR kij."""
    from thermo.interaction_parameters import IPDB

    cas_numbers = [component.cas_number for component in components]
    if not all(cas_numbers):
        return False
    for i in range(len(cas_numbers)):
        for j in range(i + 1, len(cas_numbers)):
            if not IPDB.has_ip_specific(
                "ChemSep PR",
                [cas_numbers[i], cas_numbers[j]],
                "kij",
            ):
                return False
    return True


def _component_tokens(components: list[ComponentIdentity]) -> list[set[str]]:
    return [
        {
            component.name.casefold(),
            (component.cas_number or "").casefold(),
        }
        for component in components
    ]


def matching_parameter_set(
    parameter_sets: list[ParameterSet],
    components: list[ComponentIdentity],
    model_name: str,
) -> ParameterSet | None:
    """Return the first parameter set matching component order and model name."""
    expected_tokens = _component_tokens(components)
    for parameter_set in parameter_sets:
        if parameter_set.model_name.casefold() != model_name.casefold():
            continue
        order = [token.casefold() for token in parameter_set.component_order]
        if len(order) == len(expected_tokens) and all(
            order[index] in expected_tokens[index] for index in range(len(order))
        ):
            return parameter_set
    return None


def is_srk_kij_parameter_set(parameter_set: ParameterSet) -> bool:
    return (
        parameter_set.model_name.casefold() == "srk"
        and len(parameter_set.component_order) == 2
        and parameter_set.parameter_form.casefold() in {"srk", "srk kij", "srk binary", "srk binary kij"}
        and "kij" in parameter_set.parameters
        and "kij" in parameter_set.units
    )


def has_srk_kij(
    parameter_sets: list[ParameterSet],
    components: list[ComponentIdentity],
) -> bool:
    parameter_set = matching_parameter_set(parameter_sets, components, "SRK")
    return parameter_set is not None and is_srk_kij_parameter_set(parameter_set)


def is_rk_kij_parameter_set(parameter_set: ParameterSet) -> bool:
    return (
        parameter_set.model_name.casefold() == "rk"
        and len(parameter_set.component_order) == 2
        and parameter_set.parameter_form.casefold() in {"rk", "rk kij", "rk binary", "rk binary kij"}
        and "kij" in parameter_set.parameters
        and "kij" in parameter_set.units
    )


def has_rk_kij(
    parameter_sets: list[ParameterSet],
    components: list[ComponentIdentity],
) -> bool:
    parameter_set = matching_parameter_set(parameter_sets, components, "RK")
    return parameter_set is not None and is_rk_kij_parameter_set(parameter_set)


def _required_parameter(parameters: dict[str, float], keys: tuple[str, ...]) -> float:
    normalized = {key.casefold(): value for key, value in parameters.items()}
    for key in keys:
        if key.casefold() in normalized:
            return normalized[key.casefold()]
    raise ValueError(f"Missing one of {keys} in the parameter set.")


def _coefficient_parameters(parameters: dict[str, float], prefix: str) -> list[float]:
    """Build a six-slot thermo coefficient array from constant or a+b/T keys."""
    normalized = {key.casefold(): value for key, value in parameters.items()}
    a_key = f"{prefix}_a"
    b_key = f"{prefix}_b"
    if a_key in normalized or b_key in normalized:
        if a_key not in normalized or b_key not in normalized:
            raise ValueError(f"{prefix} coefficient form requires both {prefix}_a and {prefix}_b.")
        return [normalized[a_key], normalized[b_key], 0.0, 0.0, 0.0, 0.0]
    return [_required_parameter(parameters, (prefix,)), 0.0, 0.0, 0.0, 0.0, 0.0]


def parameter_set_to_backend_params(
    parameter_set: ParameterSet,
    model_name: str,
    component_names: list[str],
) -> dict[str, object]:
    """Convert a reviewed ParameterSet into the backend gamma coefficient format."""
    if parameter_set.model_name.casefold() != model_name.casefold():
        raise ValueError(f"Parameter set model {parameter_set.model_name!r} does not match {model_name!r}.")
    names = [name.casefold() for name in component_names]
    order = [name.casefold() for name in parameter_set.component_order]
    if len(names) != 2 or order != names:
        raise ValueError("Only binary component parameter sets can be converted.")

    parameters = parameter_set.parameters
    form = parameter_set.parameter_form.casefold()
    tau = np.zeros((2, 2, 6))
    if form in {
        "nrtl",
        "nrtl binary",
        "nrtl a+b/t binary",
        "nrtl a-b",
        "nrtl coefficient",
    }:
        tau[0, 1] = _coefficient_parameters(parameters, "tau12")
        tau[1, 0] = _coefficient_parameters(parameters, "tau21")
        alpha = parameters.get("alpha", 0.3)
        alpha_coeffs = np.zeros((2, 2, 2))
        alpha_coeffs[0, 1, 0] = alpha
        alpha_coeffs[1, 0, 0] = alpha
        return {"tau_coeffs": tau, "alpha_coeffs": alpha_coeffs}
    if form in {
        "uniquac",
        "uniquac binary",
        "uniquac exp(a+b/t) binary",
        "uniquac a-b",
        "uniquac coefficient",
    }:
        tau[0, 1] = _coefficient_parameters(parameters, "tau12")
        tau[1, 0] = _coefficient_parameters(parameters, "tau21")
        return {
            "tau_coeffs": tau,
            "rs": np.asarray(
                [
                    _required_parameter(parameters, ("r1", "r_1")),
                    _required_parameter(parameters, ("r2", "r_2")),
                ],
                dtype=float,
            ),
            "qs": np.asarray(
                [
                    _required_parameter(parameters, ("q1", "q_1")),
                    _required_parameter(parameters, ("q2", "q_2")),
                ],
                dtype=float,
            ),
        }
    if form in {
        "wilson",
        "wilson binary",
        "wilson exp(a+b/t) binary",
        "wilson a-b",
        "wilson coefficient",
    }:
        lambda_coeffs = np.zeros((2, 2, 6))
        lambda_coeffs[0, 1] = _coefficient_parameters(parameters, "lambda12")
        lambda_coeffs[1, 0] = _coefficient_parameters(parameters, "lambda21")
        return {
            "Lambda_coeffs": lambda_coeffs,
            "volumes": np.asarray(
                [
                    _required_parameter(parameters, ("v1", "v_1")),
                    _required_parameter(parameters, ("v2", "v_2")),
                ],
                dtype=float,
            ),
        }
    if form in {"srk", "srk kij", "srk binary", "srk binary kij"}:
        return {"kij": _required_parameter(parameters, ("kij",))}
    if form in {"rk", "rk kij", "rk binary", "rk binary kij"}:
        return {"kij": _required_parameter(parameters, ("kij",))}
    raise ValueError(f"Unsupported parameter_form {parameter_set.parameter_form!r}.")
