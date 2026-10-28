import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'objective': 'binary:logistic',  # ou 'multi:softprob'
        'eval_metric': 'logloss',
        'random_state': 42,
        
        # PARAMÈTRE CLÉ POUR LES 1000 COLONNES :
        'tree_method': 'hist',           # Accélère drastiquement le calcul des splits
        #'device': 'cuda',              # Décommentez cette ligne si vous avez un GPU Nvidia !

        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7), # Garder des arbres peu profonds car bcp de colonnes
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7), # Force à ignorer les colonnes redondantes
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3 folds pour aller plus vite au début
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        
        # Entraînement avec Early Stopping pour couper court aux essais inefficaces
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores)

# Lancement d'une étude courte (ex: 15 essais pour tester)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print(f"Meilleure Accuracy moyenne : {study.best_value:.4f}")
print("Meilleurs hyperparamètres :", study.best_params)
