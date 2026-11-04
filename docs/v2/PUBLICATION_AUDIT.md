# Publication audit

Audit date: 2026-11-03  
Scope: recruiter-facing repository, default API/UI, CI, containers, evidence claims, secrets, licensing, and reproducibility  
Branch audited: `v2-rebuild` at and after `b32d98b`

## Release decision

**Not ready to call published yet.** The local implementation is materially stronger than the public `main` branch, but the release remains blocked until this branch is pushed, reviewed, merged, and the hosted frontend is redeployed and checked. Local verification can close code-level concerns; it cannot prove an external deployment that does not yet exist.

## Findings

| Severity | Finding | Recruiter interpretation | Status |
|---|---|---|---|
| Critical | The hosted landing page claimed real production-database predictions, deployed models, 90.2% accuracy, 0.26% MAPE, and future 2026 output from preserved static v1 data. | Metrics and deployment status appear fabricated or internally inconsistent. | Fixed locally: the publication app now exposes only the governed historical M5 fixture and explicitly disclaims live inference. Requires merge, redeploy, and live verification. |
| Critical | `v2-rebuild` has no remote tracking branch; public `main` remains at `605d962`. | The portfolio visible to a recruiter does not contain the completed rebuild. | Open release blocker. Push/PR/merge requires owner authorization and is intentionally not performed by this audit. |
| High | The top-level README described only Milestone 1, said M5 metrics were pending, said the default app served v1, and listed completed modeling work as future work. | The project looks abandoned and the documented architecture cannot be trusted. | Fixed locally with concise DS and ML-platform reading paths, measured decisions, blocked claims, and current run instructions. |
| High | CI files used deployment language for artifact-only or placeholder jobs. | Inflated MLOps claim; no actual environment promotion exists. | Fixed locally: frontend workflow is labeled as a build artifact; placeholder deploy job removed; container job claims only build/publish. |
| High | Root and backend container paths used the wrong app module or legacy health endpoint; the backend-only build context omitted v2 modules. | The advertised reproducible deployment fails on first use. | Fixed locally with one minimal v2 image, restricted dependencies, correct module, `/api/v2/health`, non-root runtime, and a narrow build context. Docker is unavailable on the audit machine, so a CI `container-smoke` job now builds the image and asserts the fail-closed contract on every push; a local serving-only venv smoke test passed, but the first real image proof lands on the pushed CI run. |
| High | Node 20 was pinned after its March 2026 end of life; Nginx 1.25 was also stale. | Unmaintained runtime and security hygiene concern. | Fixed locally: Node 24 LTS and Nginx 1.30.6; frontend container now uses `npm ci` and type-checks before build. |
| High | The v2 gitleaks action and several other CI actions still used Node 20-era releases; gitleaks v2 stopped working on hosted runners in September 2026. | The advertised secret scan can be silently absent or fail before scanning. | Fixed locally: gitleaks v3 plus Node 24-compatible checkout, setup, artifact, Docker, Trivy, and CodeQL action majors. Remote execution still requires the pushed branch. |
| High | Forty-two tracked legacy pickle files occupy about 402 MB in the worktree; the Git pack is about 243 MB. Pickle loading is also unsafe for untrusted artifacts. | Repository bloat, duplicated model lineage, and unsafe serialization risk. | Open debt. The default v2 API neither discovers nor loads them. Removal should be a separately reviewed history rewrite after a release tag/backup. |
| High | Tracked `predictions.csv` contains unsupported future 2026 forecasts and heuristic model labels. | A reviewer may mistake unvalidated output for current evidence. | Isolated from the default app and called out here. Recommended: remove it from the v2 branch in the legacy-artifact cleanup; do not cite it. |
| Medium | The frontend had nine routes containing simulated learning curves, mock notifications, scenarios, and randomized values. | A reviewer clicking beyond the forecast page finds unsupported product claims. | Fixed at the publication boundary: only `/dashboard/predictions` is routable; all other paths redirect to it. Legacy source remains for provenance. |
| Medium | Full preserved frontend lint passes only under a frozen cap and reports 193 warnings. | The build is green, but archived code quality looks unfinished under deeper review. | Publication route has a strict zero-warning gate; repository-wide warnings remain open debt. Reduce them before describing the entire frontend as production quality. |
| Medium | The Python suite is strong on contracts and evaluation but CI has no Python formatter, linter, or static type gate; the frontend has no component test suite. | Tests demonstrate domain correctness more than full software-quality coverage. | Open debt. README limits the claim to the checks that actually run. |
| Medium | Databricks, remote MLflow, Tableau Public, production monitoring, rollback, scheduled retraining, and a live backend are not externally executed. | Resume language could overstate local contracts as production operations. | Correctly blocked in README/status. Do not use verbs such as “deployed,” “operated,” or “monitored in production.” |
| Medium | The previous license line described all FRED data as public domain. FRED hosts series from multiple providers with source-specific terms. | Data-governance and licensing diligence concern. | Fixed locally: MIT applies to code only; upstream data and model terms remain their own. M5 source files are not redistributed. |
| Low | The public Tableau workbook is not published and the local workbook depends on an adjacent Hyper file. | A dashboard link cannot yet substantiate the BI claim. | Honest local-only claim. The `.twb` uses the relative `retailpred_v2.hyper` filename, so moving both together preserves the connection. |

## Evidence checked

- A throwaway Python 3.11 venv containing only `requirements-serving.txt` (fastapi, pydantic, uvicorn) served `backend.main:app` against the `.dockerignore` file set: `/api/v2/health`, `/api/v2/models`, `/api/v2/demo-catalog`, and the offline-demo `POST /api/v2/forecasts` succeeded; production mode returned 409 `production_model_unavailable`.
- Default FastAPI application mounts only `/api/v2`; legacy routes require `RETAILPRED_ENABLE_LEGACY_API=true`.
- The React fixture and backend fixture are guarded by a parity test.
- The Tableau workbook uses relative Hyper connections and five governed sheets.
- No tracked real `.env` file is present, and the local reachable history query did not find `project_root/.env`; CI still runs a full-history gitleaks scan.
- Raw M5 files, local databases, MLflow state, dbt targets, logs, virtual environments, and secrets are ignored.
- The public source branch remains `origin/main`; local work is not recruiter-visible.

## Publication checklist

- [x] `make publication-check` passes.
- [x] `make verify` passes on Python 3.11 and Node 24 LTS.
- [x] dbt parse passes from a clean environment.
- [ ] CI builds and smoke-tests the minimal API container. (The `container-smoke` job is added; proof lands on the first pushed run.)
- [ ] Push `v2-rebuild`, open a PR, review the full diff, and merge to `main`.
- [ ] Confirm branch protection and all required checks pass on the merge commit.
- [ ] Redeploy the static frontend and verify that `/` lands on the historical evidence page.
- [ ] Confirm the live page contains no production-database, deployed-model, future-forecast, accuracy, or confidence-interval claims.
- [ ] Publish Tableau only if desired, then add the verified URL; otherwise keep the local-only statement.
- [ ] Decide whether to rewrite history to remove legacy pickles and future prediction artifacts.
- [ ] Draft résumé bullets only after the public commit and links are stable.

## Safe résumé boundary

Supported language includes “implemented,” “benchmarked,” “contract-tested,” “built locally,” and “designed a fail-closed serving contract.” Unsupported language currently includes “deployed to Databricks,” “productionized,” “served live forecasts,” “operated monitoring,” “automated retraining,” and “published Tableau dashboard.”
