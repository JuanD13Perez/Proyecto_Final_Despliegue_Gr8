from feature_engine.selection import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from model.config.core import config
from model.processing import features as pp

heart_pipe = Pipeline(
    [
        # Drop features 
        #("drop_features", 
        # DropFeatures(
        #     features_to_drop=[config.model_config.temp_features]
        #     )
        #),
        # Mappers
        #(
        #    "mapper_qual",
        #    pp.Mapper(
        #        variables=config.model_config.qual_vars,
        #        mappings=config.model_config.qual_mappings,
        #    ),
        #),
        # Scaler
        ("scaler", StandardScaler()
         ),
        # XGBoost 
        ("classifier",
            XGBClassifier(
                learning_rate = config.model_config.learning_rate, 
                max_depth = config.model_config.max_depth,
                use_label_encoder= config.model_config.use_label_encoder,
                random_state = config.model_config.random_state,
                eval_metric= config.model_config.eval_metric,
            ),
        ),
    ]
)
