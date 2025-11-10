import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from mlxtend.plotting import plot_confusion_matrix
import joblib
import time
import os

# ===========================================================
# 1. Fungsi: Load Dataset
# ===========================================================
def load_dataset(path, samples_per_class=500, seed=42):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")

    # Sampling 500 per kelas (1-26)
    sampled_df = pd.DataFrame()
    for label in range(1, 27):
        subset = df[df.iloc[:, 0] == label]
        if len(subset) < samples_per_class:
            print(f"⚠️ Warning: Kelas {label} hanya memiliki {len(subset)} sampel, gunakan semuanya.")
            sampled = subset
        else:
            sampled = subset.sample(n=samples_per_class, random_state=seed)
        sampled_df = pd.concat([sampled_df, sampled])

    print(f"Sampled subset: {sampled_df.shape}")
    labels = sampled_df.iloc[:, 0].values
    images = sampled_df.iloc[:, 1:].values
    return images, labels


# ===========================================================
# 2. Fungsi: Ekstraksi HOG Feature
# ===========================================================
def extract_hog_features(images):
    hog_features = []
    print("Extracting HOG features...")
    for i in tqdm(range(len(images))):
        img = images[i].reshape(28, 28)
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(feature)
    hog_features = np.array(hog_features)
    print(f"HOG features shape: {hog_features.shape}")
    return hog_features


# ===========================================================
# 3. Fungsi: Training & Evaluasi LOOCV (dengan checkpoint)
# ===========================================================
def run_loocv(X, y, save_file="y_results_partial_RBF.csv"):
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    cv = LeaveOneOut()
    splits = list(cv.split(X))

    # Resume checkpoint jika ada
    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
        start_index = len(df)
        y_true_all = df["y_true"].tolist()
        y_pred_all = df["y_pred"].tolist()
        print(f"Resuming from index {start_index}/{len(splits)} ...")
    else:
        start_index = 0
        y_true_all, y_pred_all = [], []
        print("Starting LOOCV from scratch...")

    # Proses LOOCV
    for i in tqdm(range(start_index, len(splits)), total=len(splits), initial=start_index, desc="LOOCV Progress"):
        train_ix, test_ix = splits[i]
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        svm_clf.fit(X_train, y_train)
        y_pred = svm_clf.predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

        # Simpan tiap 100 iterasi (checkpoint)
        if (i + 1) % 100 == 0:
            pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(save_file, index=False)
            print(f"Checkpoint saved at iteration {i+1}")

    # Simpan akhir
    pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(save_file, index=False)
    print("LOOCV completed & results saved.")
    return np.array(y_true_all), np.array(y_pred_all)


# ===========================================================
# 4. Fungsi: Evaluasi & Visualisasi
# ===========================================================
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n=== Model Performance ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_names = [chr(i) for i in range(65, 91)]

    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                    class_names=class_names,
                                    figsize=(12, 10))
    plt.title("Confusion Matrix - LOOCV")
    plt.show()

    # Normalized Confusion Matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_norm,
                                    class_names=class_names,
                                    figsize=(12, 10),
                                    show_normed=True)
    plt.title("Normalized Confusion Matrix - LOOCV")
    plt.show()

    return acc, prec, rec, f1


# ===========================================================
# 5. MAIN PROGRAM
# ===========================================================
if __name__ == "__main__":
    start_time = time.time()

    # --- Load & Sampling ---
    img, lbl = load_dataset(
        r"C:\Users\Hp\Downloads\archive\ATS Machine Vision\emnist-letters-train.csv"
    )

    # --- Ekstraksi Fitur HOG ---
    hog_feat = extract_hog_features(img)

    # --- Normalisasi ---
    scaler = StandardScaler()
    X = scaler.fit_transform(hog_feat)
    y = lbl

    # --- SVM + LOOCV ---
    y_true, y_pred = run_loocv(X, y)

    # --- Evaluasi ---
    acc, prec, rec, f1 = evaluate_model(y_true, y_pred)

    # --- Simpan hasil akhir ---
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    }).to_csv("final_predictions_RBF_LOOCV.csv", index=False)

    # Simpan model & scaler
    joblib.dump(scaler, "scaler_hog.pkl")
    joblib.dump(SVC(kernel='rbf', C=1.0, gamma='scale'), "svm_rbf_model.pkl")

    print(f"\nTotal waktu eksekusi: {(time.time() - start_time)/60:.2f} menit")