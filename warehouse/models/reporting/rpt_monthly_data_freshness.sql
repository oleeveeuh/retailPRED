with latest_retail as (
    select
        'retail' as dataset,
        max(observation_date) as latest_observation_date,
        max(ingested_at) as latest_ingestion_time
    from {{ ref('stg_monthly_observations') }}
),
latest_economic as (
    select
        'economic' as dataset,
        max(observation_date) as latest_observation_date,
        max(ingested_at) as latest_ingestion_time
    from {{ ref('stg_economic_observations') }}
)

select * from latest_retail
union all
select * from latest_economic
