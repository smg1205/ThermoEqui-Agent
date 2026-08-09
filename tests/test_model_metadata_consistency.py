"""Ensure model cards and the model catalog never drift apart."""

from __future__ import annotations

from agent.router import load_model_cards
from thermo_engine.model_catalog import load_model_catalog


def test_every_catalog_model_has_a_matching_card() -> None:
    catalog = load_model_catalog()
    cards = {card.model_name: card for card in load_model_cards()}

    assert set(catalog) == set(cards)


def test_model_cards_match_catalog_metadata() -> None:
    catalog = load_model_catalog()
    cards = {card.model_name: card for card in load_model_cards()}

    for name, entry in catalog.items():
        card = cards[name]
        assert card.family == entry.family
        assert card.supported_tasks == entry.supported_equilibrium_types
        assert card.excluded_systems == entry.excluded_systems
        assert card.requires_binary_parameters == entry.requires_binary_parameters
        assert card.pressure_regime == entry.pressure_regime
        assert card.validation_requirements == entry.validation_requirements
        assert card.implementation_status == entry.implementation_status
        assert card.production_ready == entry.production_ready
