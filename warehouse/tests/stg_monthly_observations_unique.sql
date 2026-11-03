select series_id, observation_date, vintage_time
from {{ ref('stg_monthly_observations') }}
group by series_id, observation_date, vintage_time
having count(*) > 1
