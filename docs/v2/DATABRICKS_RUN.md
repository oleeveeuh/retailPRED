# Databricks execution evidence

Live claims live here. Nothing below has executed on a Databricks workspace
yet; every entry stays at its current claim level until a real run records
evidence, and the level never silently upgrades.

| Capability | Claim level | Evidence required to upgrade |
|---|---|---|
| Table contracts and Delta binding | locally verified (`make spark-check` passes on a local delta-spark session) | workspace bootstrap run: job URL, table list |
| Bundle deploy (bootstrap job, scheduled pipeline) | implemented | `databricks bundle deploy -t dev` output; job links |
| Census ingestion task | implemented (first keyed run hit an API schema mismatch; diagnostic captured) | a successful run: row counts, bronze/silver manifest hashes |
| Scoring task | implemented; refuses while the candidate is an unauthorized challenger | a post-promotion run: forecast row counts, benchmark statuses |
| Realized-error task | implemented | first matured-target evaluation rows |
| Snapshot publication to `live-data` | implemented | the data-branch commit with its pipeline run id |
| Recurring schedule | declared paused | unpause commit + first scheduled-run URL, then `is_live=true` |
| GitHub Pages retrieval | implemented (workflow defined) | first Pages deployment URL + frontend fetch check |

## Run log

(append one row per real run; never delete rows)

| Date | Job / task | Result | Link or run id |
|---|---|---|---|
| — | — | — | — |
