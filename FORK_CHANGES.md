# Fork changes

## 2026-08-29 - MySR 0.2.0

- Renamed the MySR-owned public and implementation regressor class to
  `MySRRegressor` throughout runtime APIs, annotations, CLI behavior, examples, and
  tests. No public `PySRRegressor` compatibility alias is retained; truthful PySR
  names remain in provenance, licensing, upstream links, and benchmark identities.
- Added an opt-in AI Feynman-inspired automatic input feature-engineering stage with
  independent `suggest` and `augment` modes. It runs before feature selection and
  denoising, and replays accepted transforms during prediction.
- Added fixed and learned pairwise generalized-symmetry candidates, including
  `xi±a*xj`, `xi*xj^a`, and `xi/xj^a`, plus bounded multi-level composition search.
- Added an MLP surrogate ensemble, multiple perturbation scales, separate surrogate
  training/candidate-construction/validation splits, importance-prioritized pair
  scheduling, domain checks, stability filtering, and detailed diagnostic reports.
- Recorded the AI Feynman papers and MIT-licensed reference implementation as
  algorithmic inspiration. MySR does not copy, bundle, or import the official AI
  Feynman source code.
- The feature-engineering switch remains off by default. Version 0.2.0 supports only
  single-output empirical mode in this branch; dimensional theoretical and
  semi-theoretical modes, FEAT, and the complete AI Feynman solver remain deferred.

## 2026-08-27 - Pinned MySRCore release resolution

- Changed the production JuliaPkg source from a sibling development path to the
  MySRCore.jl GitHub repository pinned at `v0.1.0`.
- Updated the development helper to override the `MySRCore` package with a local
  checkout while preserving its UUID and preferences.
- Added configuration contract tests and documented separate installation and
  two-repository development workflows.

## 2026-08-26 - MySR 0.1.0 package foundation

- Renamed the Python distribution and import package from `pysr` to `mysr`.
- Renamed the MySR-owned public and implementation regressor class to
  `MySRRegressor` without retaining a public `PySRRegressor` alias. Upstream PySR
  provenance remains documented separately.
- Moved the Julia backend into the independent `MySRCore.jl` repository and changed
  `juliapkg.json` to use that local development package.
- Replaced the incorrect root MIT metadata with the upstream Apache-2.0 license.
- Added MySR ownership, upstream provenance, and independent-fork notices.

Algorithm implementation files retained from PySR remain attributed to PySR unless a
later entry identifies a MySR-specific modification.
