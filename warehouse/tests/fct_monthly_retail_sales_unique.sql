select series_id, observation_date
from {{ ref('fct_monthly_retail_sales') }}
group by series_id, observation_date
having count(*) > 1
