#!/usr/bin/env python3
"""
Advanced analytics: clustering and a simple predictive model.
Saves cluster assignments, PCA plots, silhouette scores, and a RandomForest classifier
that predicts 'high rating' (top 25%).

Usage:
  analysis/venv/bin/python3 analysis/advanced.py --input analysis/merged_full.csv --out_dir analysis/advanced_outputs
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

sns.set(style='whitegrid')


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', default='analysis/advanced_outputs')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    figs = os.path.join(args.out_dir, 'figures')
    ensure_dir(figs)

    df = pd.read_csv(args.input)

    # Select numeric features we care about
    candidate = ['rating','average_combat_score','average_damage_per_round','kills_per_round','assists_per_round','headshot_percentage','rounds_played','kills','deaths','assists','first_kills','first_deaths']
    features = [c for c in candidate if c in df.columns]
    if len(features) == 0:
        print('No features available for clustering/modeling; aborting')
        return

    # Prepare dataset: drop rows missing any feature
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    print('Rows available for analysis:', len(data))

    # Convert headshot_percentage if present but stored as fraction or percent
    if 'headshot_percentage' in data.columns:
        # If values appear >1, assume percent like 25 -> 0.25
        if data['headshot_percentage'].max() > 1.0:
            data['headshot_percentage'] = data['headshot_percentage'] / 100.0

    # limit dataset sizes for heavy operations
    n_rows = len(data)
    sample_for_sil = data.sample(n=min(20000, n_rows), random_state=0)

    # scaling
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(sample_for_sil)

    # find best k by silhouette (2..8)
    silhouettes = {}
    for k in range(2,9):
        km = KMeans(n_clusters=k, random_state=0, n_init='auto')
        labels = km.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        silhouettes[k] = sil
    sil_df = pd.Series(silhouettes).sort_index()
    sil_df.to_csv(os.path.join(args.out_dir, 'silhouette_by_k.csv'))

    # choose best k
    best_k = int(sil_df.idxmax())

    # Fit KMeans on a larger sample (up to 100k) to assign clusters
    sample_for_k = data.sample(n=min(100000, n_rows), random_state=1)
    X_k = scaler.fit_transform(sample_for_k)
    km = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
    labels_k = km.fit_predict(X_k)
    sample_for_k['cluster'] = labels_k

    # PCA for visualization
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_k)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels_k, palette='tab10', s=10, linewidth=0)
    plt.title(f'KMeans clusters (k={best_k}) - PCA projection')
    plt.tight_layout()
    plt.savefig(os.path.join(figs, 'kmeans_pca.png'), dpi=150)
    plt.close()

    # cluster summary
    cluster_summary = sample_for_k.groupby('cluster')[features].agg(['mean','std','count'])
    cluster_summary.to_csv(os.path.join(args.out_dir, 'cluster_summary.csv'))

    # Save cluster centers (inverse transform)
    centers = km.cluster_centers_
    try:
        centers_orig = scaler.inverse_transform(centers)
    except Exception:
        centers_orig = centers
    centers_df = pd.DataFrame(centers_orig, columns=features)
    centers_df.to_csv(os.path.join(args.out_dir, 'kmeans_centers.csv'), index_label='cluster')

    # Save sample cluster assignments
    sample_for_k.to_csv(os.path.join(args.out_dir, 'clustered_sample.csv'), index=False)

    # Save silhouette plot
    plt.figure(figsize=(6,4))
    sil_df.plot(marker='o')
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.tight_layout()
    plt.savefig(os.path.join(figs, 'silhouette_by_k.png'), dpi=150)
    plt.close()

    # ===== Predictive model: predict high rating =====
    data_all = data.copy()
    # Define high rating as top 25% within available rows
    thresh = data_all['rating'].quantile(0.75)
    data_all['high_rating'] = (data_all['rating'] >= thresh).astype(int)

    # features for model
    X = data_all.drop(columns=['rating','high_rating']) if 'rating' in data_all.columns else data_all.drop(columns=['high_rating'])
    y = data_all['high_rating']

    # sample for modeling to keep runtime reasonable
    model_sample = data_all.sample(n=min(200000, len(data_all)), random_state=2)
    Xm = model_sample.drop(columns=['rating','high_rating']) if 'rating' in model_sample.columns else model_sample.drop(columns=['high_rating'])
    ym = model_sample['high_rating']

    # scale
    scaler_m = StandardScaler()
    Xm_scaled = scaler_m.fit_transform(Xm)

    X_train, X_test, y_train, y_test = train_test_split(Xm_scaled, ym, test_size=0.2, random_state=0, stratify=ym)

    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(args.out_dir, 'classification_report.csv'))

    auc = roc_auc_score(y_test, y_proba)
    # save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, 'roc_curve.png'), dpi=150)
    plt.close()

    # feature importances
    fi = pd.Series(clf.feature_importances_, index=Xm.columns).sort_values(ascending=False)
    fi.to_csv(os.path.join(args.out_dir, 'feature_importances.csv'))
    plt.figure(figsize=(8,6))
    sns.barplot(x=fi.values[:20], y=fi.index[:20])
    plt.title('Top feature importances')
    plt.tight_layout()
    plt.savefig(os.path.join(figs, 'feature_importances.png'), dpi=150)
    plt.close()

    # Save model and scalers
    joblib.dump(clf, os.path.join(args.out_dir, 'rf_high_rating.joblib'))
    joblib.dump(scaler_m, os.path.join(args.out_dir, 'scaler_model.joblib'))
    joblib.dump(km, os.path.join(args.out_dir, 'kmeans_model.joblib'))

    # Write summary
    with open(os.path.join(args.out_dir, 'advanced_report.txt'), 'w') as f:
        f.write(f'Rows used for modeling: {len(model_sample)}\n')
        f.write(f'Rating threshold (75th percentile): {thresh}\n')
        f.write(f'Best k (silhouette): {best_k} (score={silhouettes[best_k]:.4f})\n')
        f.write('\nClassification AUC: {:.4f}\n'.format(auc))
        f.write('\nTop features:\n')
        f.write('\n'.join([f'{i}. {name}: {val:.6f}' for i,(name,val) in enumerate(fi.items(), start=1)]))

    print('Advanced analytics complete. Outputs in', args.out_dir)

if __name__ == '__main__':
    main()
