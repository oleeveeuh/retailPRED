# RetailPRED v2 Decision Log

Decisions are recorded before implementation so later changes can distinguish deliberate architecture from accidental drift.

## D001: Preserve v1 instead of rewriting it in place

- Status: accepted
- Decision: tag revision `605d962` as `v1-undergrad` and implement v2 on `v2-rebuild`.
- Reason: the project’s value includes visible growth from the original implementation, and v1 metrics must remain auditable.

## D002: Standardize local and CI Python on 3.11

- Status: accepted
- Decision: use `.python-version`, a Python 3.11 virtual environment, and `requirements-dev.lock`.
- Reason: the existing backend and CI already target 3.11, while the prior local test environment was unintentionally using 3.14.

## D003: Separate macro and operational forecasting

- Status: accepted
- Decision: treat MRTS and M5 as separate use cases sharing platform contracts.
- Reason: monthly national retail categories and store/item demand differ in scale, hierarchy, covariates, and business interpretation.

## D004: Use four nonredundant model families

- Status: accepted
- Decision: Seasonal Naive, global LightGBM, NHITS, and Chronos-2 are the v2 suite.
- Reason: each represents a distinct modeling paradigm. Historical models remain documented but leave the primary interface.

## D005: Fail explicitly instead of silently simulating production output

- Status: accepted
- Decision: unavailable data, models, or explanations return typed unavailable/error states.
- Reason: randomized and heuristic fallbacks conceal operational failures and weaken result credibility.

## D006: Freeze legacy frontend lint debt without allowing growth

- Status: accepted, temporary
- Decision: convert the audited legacy violations to warnings and cap `npm run lint` at 215 warnings.
- Reason: this makes lint a blocking CI check immediately. The cap must decline as v2 replaces affected components; `npm run lint:strict` remains the target gate.

## D007: Default orchestration direction

- Status: proposed
- Decision: prefer Databricks Workflows for the lakehouse pipeline unless a demonstrated portability requirement justifies Airflow.
- Reason: maintaining two orchestrators would add complexity without improving the portfolio evidence.
