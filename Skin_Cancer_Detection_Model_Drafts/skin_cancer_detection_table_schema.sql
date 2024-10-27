skin_cancer_detection_table_schema
CREATE TABLE skin_cancer_data (
    lesion_id VARCHAR(50),
    image_id VARCHAR(50) PRIMARY KEY,
    dx VARCHAR(50),
    dx_type VARCHAR(50),
    age FLOAT,
    sex VARCHAR(10),
    localization VARCHAR(50),
    image_data BYTEA
);