# Fork changes

## 2026-09-11 - MySR 1.1.3

- Refined AFE feature-seed exploration and replay diversity, adding reciprocal and
  normalized-sub operators by default and expanding residual-complementary beam breadth.
- Strengthened RNN-GPSR proposal path with length-aware replay ranking and
  deterministic fallback fill to avoid underfilled proposal batches.
- Raised RNN-GPSR default search budgets (candidate count, proposal count,
  rounds) to better expose high-compute potential on long-budget runs.
- Updated release metadata to `v1.1.3`/`MySRCore` `v1.1.3` for synchronized
  deployment.

## 2026-09-08 - MySR 1.1.2

- Published the patch release paired with MySRCore 1.1.2 for the corrected
  benchmark scoring and capability ablation protocol.
- Kept the 1.1.1 RNN-GPSR length-alignment behavior and exposed no new search
  defaults; the release isolates the benchmark fixes from algorithm changes.

## 2026-09-02 - MySR 1.1.0

- Added formula_type-conditioned RNN-GPSR proposals with a PyTorch RNN policy,
  MySRCore dimensional validation, lightweight backend GPSR, and elite feedback
  across configurable rounds.
- Routed AI Feynman-inspired generated features into the RNN-GPSR path when both
  features are enabled, while preserving direct user `guesses` as highest-priority
  formal-search seeds.
- Added configurable RNN-GPSR parameters, batched proposal sampling, bootstrap
  structural training, and focused dimensional/seeding regression tests.
- Pinned the Python frontend to the matching MySRCore.jl `v1.1.0` release.

## 2026-09-01 - MySR 1.0.0

- Promoted the MySRCore-backed dimensional search and formula-type contract to the
  first stable MySR release.
- Added formula-type-aware AI Feynman-inspired and FEAT-like feature engineering,
  including replayable AST expressions, dimension metadata, and user-configurable
  variable/operator complexity.
- Removed the inherited `X_units`, `y_units`, `dimensional_constraint_penalty`, and
  `dimensionless_constants_only` interfaces; the public contract now uses explicit
  `X_dimensions` and `y_dimensions` only.
- Pinned the Python frontend to the matching MySRCore.jl `v1.0.0` release.

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
## 2026-09-07 - MySR 1.1.1

- Fixed RNN-GPSR teacher-forcing grammar-mask dimensions when the active batch's
  longest expression is shorter than the configured maximum expression length.
- Added regression coverage for mixed short sequences and aligned the pinned
  MySRCore backend to `v1.1.1`.
