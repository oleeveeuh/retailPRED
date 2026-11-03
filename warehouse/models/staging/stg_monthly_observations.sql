select
    cast(series_id as string) as series_id,
    cast(observation_time as date) as observation_date,
    cast(value as double) as value,
    cast(release_time as timestamp) as release_time,
    cast(vintage_time as timestamp) as vintage_time,
    cast(ingested_at as timestamp) as ingested_at,
    cast(source as string) as source
from {{ source('silver', 'monthly_observations') }}
