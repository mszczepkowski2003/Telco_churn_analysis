
-- Data base will follow the basic star schema where the core entity will be the demographics table 
-- containing informations about specific customers
create table demographics(
	customer_id varchar(50) primary key,
	gender varchar(10),
	age integer,
	senior varchar(10),
	married varchar(10),
	dependents varchar(10),
	number_of_dependents integer
);

create table location(
	customer_id varchar(50) primary key,
	location_id varchar(50),
	country varchar(100),
	state varchar(100),
	city varchar(100),
	zip_code char(5),
	lat_long varchar(100)
	);
	
create table churn_service(
	customer_id varchar(50) primary key, 
	service_id varchar(50),
	quarter varchar(5),
	reffered_friend Varchar(10),
	n_refferals integer,
	tenure_months integer,
	offer Varchar(100),
	phone_service Varchar(10),
	multiple_lines Varchar(10),
	interntet_service varchar(10),
	internet_type varchar(50),
	avg_monthly_gb_download integer, 
	online_security varchar(10),
	online_backup varchar(10),
	device_prot_plan varchar(10),
	premium_support varchar(10),
	streaming_tv varchar(10),
	streaming_movies varchar(10),
	streaming_music varchar(10),
	unlimited_data varchar(10),
	contract varchar(100),
	paperless_billing varchar(10),
	payment_method varchar(100),
	monthly_charge decimal(10,2), 
	total_charges decimal(10,2),
	total_refunds decimal(10,2),
	total_extra_data_charges decimal(10,2),
	total_long_dist_charges decimal(10,2), 
	total_revenue decimal(10,2)
	);

create table churn_status(
	customer_id varchar(50) primary key,
	status_id varchar(50),
	quarter varchar(5),
	satisfaction_score integer,
	customer_status varchar(50),
	churn_label varchar(10),
	churn_score integer,
	cltv integer,
	churn_category varchar(100),
	churn_reason varchar(200)
	);	

create table population(
	id integer primary key,
	zip_code char(5),
	population integer
);

-- adding data from US census
create table income(
zip_code char(5) primary key,
median_income decimal(10,2),
mean_income decimal(10,2)
);

-- saving changes made to the database 
commit;




