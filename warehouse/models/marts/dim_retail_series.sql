select distinct
    cast(series_id as string) as series_id,
    cast(source_code as string) as source_code,
    cast(series_name as string) as series_name,
    cast(unit as string) as unit,
    cast(seasonal_adjustment as string) as seasonal_adjustment,
    cast(source as string) as source
from {{ source('gold', 'monthly_series') }}
