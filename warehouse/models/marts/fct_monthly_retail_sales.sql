select
    cast(series_id as string) as series_id,
    cast(observation_time as date) as observation_date,
    cast(value as double) as sales_value,
    cast(release_time as timestamp) as release_time,
    cast(vintage_time as timestamp) as vintage_time
from {{ source('gold', 'monthly_series') }}
