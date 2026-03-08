
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from src.pipeline.transformers import Winsorizer,SpatialNeighborTransformer,FeatureEngineerOne,FeatureEngingeerTwo,IsolationForestTransformer
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

def preprocessor() -> Pipeline: 
    """Inicjalizuje i konfiguruje pełen potok przetwarzania danych (preprocessing pipeline).
    
    Kroki potoku (Pipeline steps): 
        - Zmniejszenie wymiarowości zmiennej 'city' poprzez przyporządkowanie rzadkich klas do 'Other' (Winsorizer).
        - Generowanie cech przestrzennych na bazie odległości dla 'latitude' i 'longitude' (SpatialNeighborTransformer).
        - Inżynieria cech I generująca m.in. licznik brakujących wartości i flagi refundacji (FeatureEngineerOne).
        - Rozdzielne kodowanie zmiennych (OrdinalEncoding, TargetEncoding, OneHotEncoding).
        - Imputacja warunkowa metodą K-Nearest Neighbors dla wybranych cech (knn_imputer).
        - Imputacja cech demograficznych i kategorycznych przy użyciu IterativeImputer + RandomForestRegressor (rf_imputer).
        - Inżynieria cech II kategoryzująca m.in. zaimputowany już wiek na przedziały 'age_binned' (FeatureEngingeerTwo).
        - Ordynalne zakodowanie nowych zmiennych i wykrywanie anomalii przy użyciu IsolationForest (isolation_forest).
        - Ostateczne skalowanie wyselekcjonowanych kolumn przez StandardScaler (scaler).
    
    Returns:
        Pipeline: Skonfigurowany potok Scikit-learn zdefiniowany dla transformacji i oczyszczania cech wejściowych.
    """
    # ENCODING
    ordinal_columns = ['senior','married','dependents','reffered_friend','phone_service',
                        'multiple_lines','interntet_service','online_security', 'online_backup',
                        'device_prot_plan','premium_support', 'streaming_tv','streaming_movies',
                        'streaming_music','unlimited_data','paperless_billing','contract']
    
    ordinal_categories = [['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],
                        ['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],
                        ['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],['No', 'Yes'],
                        ['No', 'Yes'],['Month-to-Month', 'One Year', 'Two Year']]

    target_encoding_columns = ['city','offer','internet_type']

    ohe_encoding_columns = ['payment_method']

    rf_imputer_columns = ['senior', 'avg_monthly_gb_download', 'monthly_charge', 'internet_type','dependents','age_NA']

    #IMPUTATION  
    knn_imp_columns = ['latitude','longitude','median_income']
    rf_imp_columns = ['senior','avg_monthly_gb_download','monthly_charge','internet_type','interntet_service','dependents','reffered_friend', 'streaming_movies','premium_support','age_NA']

    # POWER TRANSFORMER
    pt_cols = ['avg_monthly_gb_download', 'monthly_charge','latitude','longitude','median_income',
    'n_refferals','total_charges', 'total_refunds'
        , 'total_long_dist_charges','neighbors_within_10km','number_of_dependents','total_extra_data_charges','zip_population']

    ## STANDARD SCALER 

    st_cols= ['internet_type', 'dependents', 'reffered_friend',
        'premium_support', 
        'multiple_lines', 'online_security', 'online_backup',
        'device_prot_plan', 'streaming_tv', 'streaming_movies',
        'paperless_billing', 'contract',
        'payment_method_Bank Withdrawal', 'payment_method_Credit Card',
        'payment_method_Mailed Check', 
        'tenure_months', 
            'cltv', 'missing_age_NA',
        'missing_median_income', 'refund_present','city','offer','outlier_label','age_NA','premium_services'] 
    all_st_cols = st_cols + pt_cols
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

    Categorical_encoder = ColumnTransformer([('ordinal_encoding', OrdinalEncoder(categories = ordinal_categories), ordinal_columns),
                                             ('target_encoding', TargetEncoder(
                                                 categories='auto'
                                                 ,target_type='binary'
                                                 ,smooth='auto'
                                                 , random_state=RANDOM_STATE
                                                 ),target_encoding_columns), 
                                            ('ohe_encoding' , OneHotEncoder(
                                                sparse_output=False
                                                , handle_unknown='warn'
                                                ), ohe_encoding_columns)],
                                            verbose_feature_names_out=False, 
                                            remainder = 'passthrough').set_output(transform = 'pandas')
    

    knn_logic = Pipeline([('scaler', StandardScaler()),
                      ('imputer',KNNImputer(n_neighbors=5, weights='distance'))
                      ]).set_output(transform='pandas')

    knn_imputer = ColumnTransformer([('knn_logic',knn_logic,knn_imp_columns)], verbose_feature_names_out=False, remainder='passthrough').set_output(transform='pandas')

    rf_imputer = ColumnTransformer([('imputer', IterativeImputer(estimator = RandomForestRegressor(n_estimators=50,
                                                                                                   random_state=RANDOM_STATE,
                                                                                                   n_jobs=-1, 
                                                                                                   max_depth=10),
                              max_iter=10,random_state=RANDOM_STATE, skip_complete=True), 
                              rf_imputer_columns)], verbose_feature_names_out=False, remainder='passthrough').set_output(transform='pandas')
    
    bin_age_encoder = ColumnTransformer([('encoder',OrdinalEncoder(), ['age_NA'])], verbose_feature_names_out=False, remainder='passthrough').set_output(transform='pandas')

    # scaler = ColumnTransformer([('standard_scaler', StandardScaler(),st_cols),
    #                         ('pt_scaler',PowerTransformer(method='yeo-johnson'),pt_cols)], verbose_feature_names_out=False, remainder='drop')
    scaler =  ColumnTransformer([('standard_scaler', StandardScaler(),all_st_cols)], verbose_feature_names_out=False, remainder='drop').set_output(transform = 'pandas')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

    preprocessor = Pipeline([('winsorize', Winsorizer(variable = 'city', treshold=20)),
                 ('snt',SpatialNeighborTransformer()), 
                 ('feature_engineering1', FeatureEngineerOne(missing_cols = ['age_NA','median_income'],check_refund=True, premium_columns=['online_security', 'online_backup', 'device_prot_plan', 'premium_support'])), 
                 ('encoder', Categorical_encoder),
                 ('knn_imputation',knn_imputer),
                 ('rf_imputation',rf_imputer),
                 ('feature_engineer2', FeatureEngingeerTwo()),
                 ('age_encoder',bin_age_encoder),
                 ('isolation_forest',IsolationForestTransformer(n_estimators = 500, contamination=0.02,random_state=RANDOM_STATE)),
                 ('scaler',scaler)
                  ])

    return preprocessor