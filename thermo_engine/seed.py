"""Command-line seeding of reviewed production parameter sets."""

from __future__ import annotations

import argparse
import os

from database.session import Repository, create_database_engine, initialize_database
from thermo_engine.parameter_store import (
    load_production_parameter_sets,
    seed_production_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed reviewed production parameter sets into the ThermoEqui database."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy database URL; defaults to DATABASE_URL or the local SQLite file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate production YAML without writing to the database.",
    )
    args = parser.parse_args()

    if args.check_only:
        parameter_sets = load_production_parameter_sets()
        print(f"Validated {len(parameter_sets)} production parameter sets.")
        return

    engine = create_database_engine(args.database_url)
    initialize_database(engine)
    repository = Repository(engine)
    result = seed_production_parameters(repository)
    print(
        "Seeded "
        f"{result.total} parameter sets "
        f"({result.added} added, {result.updated} updated, {result.unchanged} unchanged, "
        f"{result.removed} stale duplicates removed)."
    )


if __name__ == "__main__":
    main()
