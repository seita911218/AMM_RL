import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def clustering(df_features,features_list,best_km = 3,window=24*30*2,random_state=42,IS_end='2024-01-29 19:00'):
    df = df_features.copy()
    df_IS = df[df.index<=IS_end].copy()
    N =len(df_IS)
    X = df_IS[features_list+['return','return_next_hr']].copy()
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    # normalize IS and OS separately
    X_proc = preprocess.fit_transform(X)
    df.loc[df.index[:N],['return_norm','return_next_hr_norm']] = X_proc[:,-2:]
    X_proc = X_proc[:,:-2]

    if not best_km :
        Ks = range(2, 11)
        kmeans_inertia, kmeans_sil = [], []

        for k in Ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X_proc)
            kmeans_inertia.append(km.inertia_)
            kmeans_sil.append(silhouette_score(X_proc, labels))

        best_km = Ks[int(np.argmax(kmeans_sil))]
        print("K for KMeans:", best_km)

    km = KMeans(n_clusters=best_km, n_init=20, random_state=random_state).fit(X_proc)
    labels_IS = km.labels_

    # rolling woindow for out of sample normalization 
    rolling_mean = df[features_list].rolling(window).mean()
    rolling_std = df[features_list].rolling(window).std()
    X_norm = (df[features_list] - rolling_mean) / rolling_std
    X_proc_OS = X_norm[X_norm.index > IS_end].copy()
    label_OS = km.predict(X_proc_OS.values)
    df["km_label"] = np.hstack([labels_IS,label_OS])
    df.loc[df.index[N:],['return_norm','return_next_hr_norm']] = X_proc_OS.values[:,-2:]


    kmm_centers = pd.DataFrame(km.cluster_centers_,columns=features_list)

    #relabel
    df_IS = df[df.index<=IS_end].copy()
    means = df_IS.groupby('km_label')['return_next_hr_norm'].mean()
    order = means.sort_values().index
    label_map = {old: new for new, old in enumerate(order)}
    df['km_label'] = df['km_label'].map(label_map).astype('int64')
    kmm_centers.index =[ label_map[i] for i in kmm_centers.index]
    
    return kmm_centers.sort_index(),df

def plot_pca(df,text,n_clusters=3,random_state=42):
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_proc = preprocess.fit_transform(df)
    pca = PCA(n_components=n_clusters, random_state=random_state).fit(X_proc)
    coords = pca.transform(X_proc)
    df["pca1"], df["pca2"] = coords[:,0], coords[:,1]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='km_label', palette='viridis', s=50, alpha=0.7)
    plt.title(f'K-Means result (PCA ),{text}')
    plt.grid(True)

def plot_cen_hm(df,km,text):
    plt.figure(figsize=(18, 10))
    R = df.groupby('km_label')[['return_norm','return_next_hr_norm']].mean()

    kmm_centers = pd.concat([km,R],axis=1 )
    sns.heatmap( kmm_centers
                    ,cmap='coolwarm',cbar=True,
                    linewidth=0.01,linecolor='k',annot=True)
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor');
    plt.title(f'Centroid of each cluster,{text}')



def compute_blocks(df, label_col='km_label'):
    idxs = [df.index[0]]
    lab = df[label_col].values
    for i in range(len(df)-1):
        if lab[i] != lab[i+1]:
            idxs.append(df.index[i])
    return idxs

def plot_one_dataset(ax_main, ax_sub, df, title,col='reward', label_col='km_label',
                     y_main=None, xlim=None):
    blocks = compute_blocks(df, label_col)
    uniq = sorted(df[label_col].unique())
    palette = sns.color_palette('Set2', n_colors=len(uniq))
    color_map = {lab: c for lab, c in zip(uniq, palette)}

    for i in range(1, len(blocks)):
        start_time = blocks[i-1]
        end_time   = blocks[i]
        lab = df.loc[start_time, label_col]
        ax_main.axvspan(start_time, end_time, color=color_map[lab], alpha=0.4)
    last_lab = df.loc[blocks[-1], label_col]
    ax_main.axvspan(blocks[-1], df.index[-1], color=color_map[last_lab], alpha=0.4)

    ax_main.plot(df.index, df[col], color='black', linewidth=1.5, label='cumPnL Trajectory')
    ax_main.set_title(title, fontsize=14)
    ax_main.set_ylabel(col, fontsize=12)
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.legend(loc='upper left')
    if xlim is not None:
        ax_main.set_xlim(*xlim)
    if y_main is not None:
        ax_main.set_ylim(*y_main)

  

    x_step = blocks + [df.index[-1]]
    y_step = [df.loc[t, 'km_label'] for t in blocks]
    y_step.append(df.loc[blocks[-1], 'km_label'])

    ax_sub.step(x_step, y_step, where='post', linewidth=1.5, color='darkblue')
    ax_sub.set_ylabel('regime', fontsize=12)
    ax_sub.set_facecolor('aliceblue')
    ax_sub.grid(True, linestyle='--', alpha=0.6)

def plot_regime_boxplot(df,text,col='return_next_hr_norm',outlier_pct=True,y_lim=(-2,2)):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x='km_label',
        y=col,
        linewidth=1,
        fliersize=3,
        showmeans=True, 
        meanprops={"marker":"D", "markerfacecolor":"red", "markeredgecolor":"black"}
    )
    plt.title(text)
    plt.xlabel('Regime Label')
    plt.ylabel(col)
    if y_lim is not None:
        plt.ylim(y_lim[0],y_lim[1])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    if outlier_pct:
        outlier_pct_dict = {}
        for label in np.sort(df['km_label'].unique()):
            group = df[df['km_label'] == label][col].copy()
            q1 = group.quantile(0.25)
            q3 = group.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = ((group < lower) | (group > upper)).sum()
            total_count = group.count()
            outlier_pct_dict[label] = outlier_count / total_count * 100

        for label, pct in outlier_pct_dict.items():
            print(f"Regime {label}: Outlier {pct:.2f}%")