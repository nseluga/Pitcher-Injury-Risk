import joblib, pandas as pd, sys, numpy as np
sys.path.insert(0, '.')
from src.models.baseline_models import _infer_feature_cols

fm = pd.read_parquet('data/processed/feature_matrix.parquet')
fc = _infer_feature_cols(fm)
pipe = joblib.load('models/baseline_xgboost_tuned.joblib')
imp = pipe.named_steps['imputer']
clf = pipe.named_steps['clf']
imp_feats = list(imp.get_feature_names_out())
importances = clf.feature_importances_
imp_df = pd.DataFrame({'feature': imp_feats, 'importance': importances}).sort_values('importance', ascending=False)
print('Top 30 XGBoost feature importances:')
print(imp_df.head(30).to_string(index=False))
print()
workload_feats = ['pitch_count', 'pitches_7d', 'days_rest', 'sl_pct', 'fb_pct', 'acwr_7_28', 'pitches_28d']
for f in workload_feats:
    rows = imp_df[imp_df['feature'] == f]
    if len(rows) > 0:
        rank = int(imp_df.index[imp_df['feature'] == f][0]) + 1
        print(f'{f}: rank={rank}, importance={rows.iloc[0]["importance"]:.4f}')
    else:
        print(f'{f}: NOT IN MODEL')
