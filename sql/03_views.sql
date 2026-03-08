
-- final step: Creating Views for quick data lookups:
-- 1)v_telcom_master - most important and relevant informations
-- 2)v_telcom_geo    - demographics/geographics data
-- 3)v_telcom_full_data   - contains all of the available variables for every customer

-- v_telcom_master
create or replace view v_telcom_master as 
select 
-- Demographics
d.customer_id, d.gender, d.age,
-- Location
l.country, l.state, l.city, l.zip_code,
-- Population 
p.population,
-- Churn service
cs.tenure_months, cs.offer, cs.contract, cs.monthly_charge, cs.total_revenue,
-- Churn status
st.satisfaction_score, st.cltv, st.churn_label, st.churn_reason
from demographics d
left join location l on d.customer_id = l.customer_id
left join population p on l.zip_code = p.zip_code
left join churn_service cs on d.customer_id = cs.customer_id
left join churn_status st on d.customer_id = st.customer_id

-- v_telcom_geo

create or replace view v_telcom_geo as 
select
d.customer_id,
d.gender,
d.age,
d.married,
l.location_id,
l.country,
l.state,
l.city,
l.zip_code,
l.latitude,
l.longitude,
p.population,
cs.total_revenue,
st.churn_label
from demographics d
left join location l on d.customer_id = l.customer_id
left join population p on l.zip_code = p.zip_code
left join churn_service cs on d.customer_id = cs.customer_id
left join churn_status st on d.customer_id = st.customer_id;

-- v_telcom_full_data

CREATE OR REPLACE VIEW v_telcom_full_data AS
SELECT 
    -- Demographics
    d.customer_id, d.gender, d.age, d.senior, d.married, d.dependents, d.number_of_dependents,
    -- Location
    l.location_id, l.country, l.state, l.city, l.zip_code, l.latitude, l.longitude,
    -- Population (Linked via zip_code)
    p.population AS zip_population,
    i.median_income, i.mean_income,
    -- Churn Service
    cs.service_id, cs.quarter, cs.reffered_friend, cs.n_refferals, cs.tenure_months, cs.offer, 
    cs.phone_service, cs.multiple_lines, cs.interntet_service, cs.internet_type, 
    cs.avg_monthly_gb_download, cs.online_security, cs.online_backup, cs.device_prot_plan, 
    cs.premium_support, cs.streaming_tv, cs.streaming_movies, cs.streaming_music, 
    cs.unlimited_data, cs.contract, cs.paperless_billing, cs.payment_method, 
    cs.monthly_charge, cs.total_charges, cs.total_refunds, cs.total_extra_data_charges, 
    cs.total_long_dist_charges, cs.total_revenue,
    -- Churn Status
    st.status_id, st.satisfaction_score, st.customer_status, st.churn_label, 
    st.churn_score, st.cltv, st.churn_category, st.churn_reason
FROM demographics d
left join location l on d.customer_id = l.customer_id
left join population p on l.zip_code = p.zip_code
left join income i on l.zip_code = i.zip_code
left join churn_service cs on d.customer_id = cs.customer_id
left join churn_status st on d.customer_id = st.customer_id;

select * from v_telcom_full_data limit 50;


commit;