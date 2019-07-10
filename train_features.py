import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, accuracy_score
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Enter data dir path', default='./LDS_features_10.pkl')
    parser.add_argument('-t', '--tpot', help='Enter True or False to indicate running TPOT', default=False)
    args = parser.parse_args()
    feature_data_file = args.data
    run_tpot = args.tpot
    df = pd.read_pickle(path=feature_data_file)
    ros = RandomOverSampler(random_state=42)
    X = np.stack(df['features'][:])
    y = df['label']
    X, y = ros.fit_resample(X,y)
    if run_tpot:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tpot = TPOTClassifier(generations=10, population_size=10, max_eval_time_mins=1, verbosity=3, periodic_checkpoint_folder="./pipeline_checkpoints")
        tpot.fit(X_train,y_train)
        print(tpot.score(x_test, y_test))
        tpot.export('tpot_best_pipeline')

    pca = PCA(n_components=3)
    kpca = KernelPCA(n_components=3, kernel='rbf')
    lda = LinearDiscriminantAnalysis(n_components = 3)

    clf = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=1, min_samples_split=9, n_estimators=100)
    clf_pca = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=1, min_samples_split=9, n_estimators=100)
    clf_kpca = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=1, min_samples_split=9, n_estimators=100)
    clf_lda = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=1, min_samples_split=9, n_estimators=100)

    X_pca = pca.fit_transform(X, y=y)
    X_kpca = kpca.fit_transform(X, y=y)
    X_lda = lda.fit_transform(X,y=y)

    pca_var_ratio = pca.explained_variance_ratio_
    lda_var_ratio = lda.explained_variance_ratio_
    kpca_var = kpca.lambdas_

    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    x_pca_train, x_pca_test, y_pca_train, y_pca_test= train_test_split(X_pca,y, test_size=0.2, random_state=42)
    x_kpca_train, x_kpca_test, y_kpca_train, y_kpca_test = train_test_split(X_kpca,y, test_size=0.2, random_state=42)
    x_lda_train, x_lda_test, y_lda_train, y_lda_test = train_test_split(X_lda,y, test_size=0.2, random_state=42)

    clf.fit(x_train, y_train)
    clf_pca.fit(x_pca_train, y_pca_train)
    clf_kpca.fit(x_kpca_train, y_kpca_train)
    clf_lda.fit(x_lda_train, y_lda_train)

    y_pred = clf.predict(x_test)
    y_pca_pred = clf_pca.predict(x_pca_test)
    y_kpca_pred = clf_kpca.predict(x_kpca_test)
    y_lda_pred = clf_lda.predict(x_lda_test)

    f_measure = f1_score(y_test, y_pred)
    f_measure_pca = f1_score(y_pca_test, y_pca_pred)
    f_measure_kpca = f1_score(y_kpca_test, y_kpca_pred)
    f_measure_lda = f1_score(y_lda_test, y_lda_pred)


    acc = accuracy_score(y_test, y_pred)
    acc_pca = accuracy_score(y_pca_test, y_pca_pred)
    acc_kpca = accuracy_score(y_kpca_test, y_kpca_pred)
    acc_lda = accuracy_score(y_lda_test, y_lda_pred)

    print("PCA v. LDA\tf_measure\taccuracy")
    print("- \t\t"+str(f_measure)+"\t\t"+str(acc))
    print("PCA  \t\t"+str(f_measure_pca)+"\t\t"+str(acc_pca))
    print("LDA  \t\t"+str(f_measure_kpca)+"\t\t"+str(acc_kpca))
    print("kPCA  \t\t"+str(f_measure_lda)+"\t\t"+str(acc_lda))
