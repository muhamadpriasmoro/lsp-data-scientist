import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, mean_squared_error, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Fungsi untuk menghitung outliers
def calculate_outliers(df):
    outlier_counts = {}
    
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_counts[column] = outlier_count
    
    return outlier_counts

# Fungsi untuk menampilkan analisis data
def analyze_data(df):
    st.header("Data Analysis Dashboard")

    if st.button('Bagian 1'):
        st.subheader("5 Baris Pertama dan Terakhir")
        st.write(df.head(5))
        st.write(df.tail(5))
        
        st.subheader("Jumlah Baris dan Kolom")
        st.write(df.shape)
        
        st.subheader("Deskripsi Statistik:")
        st.write(df.describe())
        
        # Count missing values
        null_counts = df.isnull().sum()

        # Calculate outliers
        outlier_counts = calculate_outliers(df)

        # Prepare DataFrames for display
        null_counts_df = null_counts.reset_index()
        null_counts_df.columns = ['Fitur', 'Jumlah']

        outlier_counts_df = pd.Series(outlier_counts).reset_index()
        outlier_counts_df.columns = ['Fitur', 'Jumlah']

        # Display in Streamlit
        st.write("Jumlah Data Null")
        st.dataframe(null_counts_df)

        st.write("Jumlah Outlier")
        st.dataframe(outlier_counts_df)

    if st.button('Bagian 2'):
        st.subheader("Boxplot Outliers")
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df.select_dtypes(include=[np.number]))
        st.pyplot(plt)

        st.subheader("Histogram Distribusi Fitur Numerik")
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric.hist(bins=20, figsize=(10,8))
        st.pyplot(plt)

        st.subheader("Matriks Korelasi")
        # Menghitung korelasi hanya pada kolom numerik
        corr_matrix = df_numeric.corr()
        plt.figure(figsize=(10,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    if st.button('Bagian 3'):
        st.subheader("Silhouette Score dan Plot")
        
        # Memilih hanya kolom numerik untuk KMeans
        df_numeric = df.select_dtypes(include=[np.number])
        
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_numeric)
        labels = kmeans.labels_
        
        # Menghitung silhouette score
        score = silhouette_score(df_numeric, labels)
        st.write(f"Silhouette Coefficient: {score}")
        
        # Plot silhouette
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=df_numeric.iloc[:, 0], y=df_numeric.iloc[:, 1], hue=labels, palette="Set1")
        st.pyplot(plt)

    if st.button('Bagian 4'):
        st.subheader("Linear Regression (MSE dan Plot Hasil Prediksi)")

        # Memilih kolom numerik untuk regresi linear
        df_numeric = df.select_dtypes(include=[np.number])

        # Mengasumsikan kolom terakhir adalah target (y)
        X = df_numeric.iloc[:, :-1]
        y = df_numeric.iloc[:, -1]

        # Membagi dataset untuk training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Prediksi dan MSE
        y_pred = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse}")

        # Plot hasil prediksi vs aktual
        plt.figure(figsize=(10,6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Linear Regression: Actual vs Predicted')
        st.pyplot(plt)

        st.subheader("Logistic Regression (Akurasi dan Confusion Matrix)")

        # Cek jika ada kolom kategorikal seperti 'species'
        if 'species' in df.columns:
            X = df_numeric  # Gunakan hanya fitur numerik
            y = df['species']  # Target adalah kolom kategorikal

            # Menggunakan LabelEncoder untuk mengubah target menjadi numerik
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Membagi dataset menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            # Logistic Regression
            logr_model = LogisticRegression()
            logr_model.fit(X_train, y_train)

            # Prediksi
            y_pred = logr_model.predict(X_test)

            # Menghitung akurasi
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Akurasi: {accuracy}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.write(cm)

            # Plot Confusion Matrix
            plt.figure(figsize=(10,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix for Logistic Regression')
            st.pyplot(plt)
        else:
            st.write("Kolom 'species' tidak ditemukan pada data.")

        st.subheader("K-Nearest Neighbors (Akurasi dan Confusion Matrix)")
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        st.write(f"Accuracy (KNN): {accuracy_knn}")
        
        conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
        sns.heatmap(conf_matrix_knn, annot=True, cmap="Greens", fmt="d")
        st.pyplot(plt)
        

st.title("Project Data Science")

st.markdown("""
### Panduan
1. **Run file `project_deploy.py` pada terminal** dengan perintah:
   ```bash
   streamlit run project_deploy.py
2. Upload file CSV Anda dengan menekan "browse files".
3. Jika CSV berhasil diupload, akan muncul 4 bagian tombol yang berisi data yang sudah bersih dan dianalisis dari file CSV Anda. """)

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully.")
        
        analyze_data(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
else:
    st.warning("Please upload a CSV file.")

