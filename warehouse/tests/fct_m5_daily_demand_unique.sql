select series_id, demand_date
from {{ ref('fct_m5_daily_demand') }}
group by series_id, demand_date
having count(*) > 1
