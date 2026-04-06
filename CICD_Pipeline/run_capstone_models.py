from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DROP_COLUMNS = ['strategy', 'entry_date', 'exit_date', 'pnl_price', 'profitable', 'Close', 'Low', 'High', 'Date']


def find_dataset(repo_root: Path) -> Path:
    candidates = sorted(repo_root.rglob('Clean_dataset')) + sorted(repo_root.rglob('*.csv'))
    for path in candidates:
        if path.name == 'Clean_dataset':
            return path
    for path in candidates:
        if 'clean' in path.name.lower() and path.is_file():
            return path
    raise FileNotFoundError('Could not find dataset file. Expected a file named Clean_dataset or a clean CSV.')


def build_models() -> dict[str, tuple[Pipeline, list[dict]]]:
    return {
        'logreg': (
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            [
                {'model__C': [0.1, 1.0]},
            ],
        ),
        'rf': (
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', RandomForestClassifier(random_state=42)),
            ]),
            [
                {'model__n_estimators': [100], 'model__max_depth': [None, 5]},
            ],
        ),
        'mlp': (
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(max_iter=400, random_state=42)),
            ]),
            [
                {'model__hidden_layer_sizes': [(32,), (32, 16)]},
            ],
        ),
    }


def evaluate(df: pd.DataFrame, model_name: str, pipe: Pipeline, param_grid: list[dict]) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    skipped: list[dict] = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    strategies = sorted(df['strategy'].dropna().astype(str).unique())
    x_cols = [c for c in df.columns if c not in DROP_COLUMNS]

    for strat in strategies:
        df_strat = df[df['strategy'].astype(str) == strat].copy()
        y = df_strat['profitable']
        X = df_strat[x_cols].select_dtypes(include=['number']).replace([np.inf, -np.inf], np.nan)

        reason = None
        if len(df_strat) < 20:
            reason = 'too few rows'
        elif X.shape[1] == 0:
            reason = 'no numeric features'
        elif y.nunique() < 2:
            reason = 'target has one class'
        elif y.value_counts().min() < 2:
            reason = 'one class has fewer than 2 rows'

        if reason:
            skipped.append({'model': model_name, 'strategy': strat, 'reason': reason})
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        if y_train.value_counts().min() < 3:
            skipped.append({'model': model_name, 'strategy': strat, 'reason': 'not enough samples per class for 3-fold CV'})
            continue

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=1,
            error_score='raise',
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        results.append({
            'model': model_name,
            'strategy': strat,
            'rows': int(len(df_strat)),
            'features_used': int(X.shape[1]),
            'best_params': json.dumps(grid.best_params_),
            'best_cv_f1': float(grid.best_score_),
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_f1': float(f1_score(y_test, y_pred)),
            'test_auc': float(roc_auc_score(y_test, y_prob)),
        })

    return results, skipped


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = find_dataset(repo_root)
    df = pd.read_csv(dataset_path).dropna().copy()

    out_dir = repo_root / 'artifacts'
    out_dir.mkdir(exist_ok=True)

    all_results: list[dict] = []
    all_skipped: list[dict] = []

    for model_name, (pipe, param_grid) in build_models().items():
        results, skipped = evaluate(df, model_name, pipe, param_grid)
        all_results.extend(results)
        all_skipped.extend(skipped)

    results_df = pd.DataFrame(all_results)
    skipped_df = pd.DataFrame(all_skipped)

    if not results_df.empty:
        results_df = results_df.sort_values(['model', 'test_f1'], ascending=[True, False])
        results_df.to_csv(out_dir / 'ci_model_summary.csv', index=False)
        best_df = results_df.sort_values('test_f1', ascending=False).groupby('model', as_index=False).first()
        best_df.to_csv(out_dir / 'ci_best_by_model.csv', index=False)
    else:
        pd.DataFrame(columns=['model', 'strategy']).to_csv(out_dir / 'ci_model_summary.csv', index=False)
        pd.DataFrame(columns=['model', 'strategy']).to_csv(out_dir / 'ci_best_by_model.csv', index=False)

    skipped_df.to_csv(out_dir / 'ci_skipped_strategies.csv', index=False)

    print('Saved artifacts to', out_dir)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    if not skipped_df.empty:
        print('\nSkipped strategies:')
        print(skipped_df.to_string(index=False))


if __name__ == '__main__':
    main()
