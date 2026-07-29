# Parameter evidence

Every parameter set records component order, form, units, applicability ranges, equilibrium types,
source title/identifier/type, quality, and notes. Allowed unreferenced source types are
`user_supplied`, `test_fixture`, `estimated`, and `unknown`. `test_fixture` is rejected by production
repository writes. Reversing component order must transform directional parameters explicitly.

The public schema rejects duplicate or blank component identities, empty/non-finite parameter
values, incomplete unit mappings, and non-positive or reversed applicability ranges. Literature
and database records require both a source title and identifier. Parameter-set identifiers are
immutable; a duplicate production write returns `409 duplicate_parameter_set` instead of
overwriting the original record.
