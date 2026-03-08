
-- because of some errors made while creating tables, I needed to make some changes
-- dropping and changing the primary key in the table with zipcodes 
alter table population
drop constraint population_pkey;

alter table population
add primary key (zip_code);

-- dropping redundant column id (zipcode is the primary key )
alter table population 
drop column id;

alter table location 
alter column zip_code set not null;
alter table population 
alter column zip_code set not null;



-- Creating connections beetween core table demographics and the rest of the tables 
-- tables are connected via customer id 

alter table location 
add constraint fk_location_demographics
foreign key (customer_id) references demographics(customer_id);


alter table churn_service
add constraint fk_service_demographics
foreign key (customer_id) references demographics(customer_id);

alter table churn_status 
add constraint fk_status_demographics
foreign key (customer_id) references demographics(customer_id);

alter table location
add constraint fk_population_location
foreign key (zip_code) references population(zip_code);

alter table location 
add constraint fk_income_location 
foreign key (zip_code) references income(zip_code);

-- Adding two empty columns in order to split the lat_long variable (nested: (latitude, longtitude))
alter table location 
add column latitude varchar(50),
add column longitude varchar(50);

-- Updating location table with values for two columns
update location 
set 
	latitude = split_part(lat_long, ',', 1),
	longitude = split_part(lat_long, ',', 2);

-- saving changes made to the database 
commit;


