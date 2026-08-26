# Fork changes

## 2026-08-26 - MySR 0.1.0 package foundation

- Renamed the Python distribution and import package from `pysr` to `mysr`.
- Added `MySRRegressor` as the MySR public class name while retaining the upstream
  `PySRRegressor` implementation name as a compatibility alias.
- Moved the Julia backend into the independent `MySRCore.jl` repository and changed
  `juliapkg.json` to use that local development package.
- Replaced the incorrect root MIT metadata with the upstream Apache-2.0 license.
- Added MySR ownership, upstream provenance, and independent-fork notices.

Algorithm implementation files retained from PySR remain attributed to PySR unless a
later entry identifies a MySR-specific modification.
