select
    cast(series_id as string) as series_id,
    cast(date as date) as demand_date,
    cast(units as bigint) as units,
    cast(sell_price as double) as sell_price,
    cast(item_id as string) as item_id,
    cast(dept_id as string) as department_id,
    cast(cat_id as string) as category_id,
    cast(store_id as string) as store_id,
    cast(state_id as string) as state_id,
    cast(wm_yr_wk as bigint) as retail_week_id,
    cast(weekday as string) as weekday,
    cast(event_name_1 as string) as primary_event_name,
    cast(event_type_1 as string) as primary_event_type
from {{ source('gold', 'm5_demand') }}
