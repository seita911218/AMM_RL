import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def clustering(df_features,features_list,best_km = 3,window=24*30*2,random_state=42,IS_end='2024-01-29 19:00'):
    df = df_features.copy()
    df_IS = df[df.index<=IS_end].copy()
    N =len(df_IS)
    X = df_IS[features_list+['reward','reward_next_hr']].copy()
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    # normalize IS and OS separately
    X_proc = preprocess.fit_transform(X)
    df.loc[df.index[:N],['reward_norm','reward_next_hr_norm']] = X_proc[:,-2:]
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
    df.loc[df.index[N:],['reward_norm','reward_next_hr_norm']] = X_proc_OS.values[:,-2:]


    kmm_centers = pd.DataFrame(km.cluster_centers_,columns=features_list)

    #relabel
    df_IS = df[df.index<=IS_end].copy()
    means = df_IS.groupby('km_label')['reward_next_hr_norm'].mean()
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
    reward = df.groupby('km_label')[['reward_norm','reward_next_hr_norm']].mean()

    kmm_centers = pd.concat([km,reward],axis=1 )
    sns.heatmap( kmm_centers
                    ,cmap='coolwarm',cbar=True,
                    linewidth=0.01,linecolor='k',annot=True)
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor');
    plt.title(f'Centroid of each cluster,{text}')