import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import svd, eigvals, eig

st.title('SIMPLE VECTOR MATRIX APPS')

with st.sidebar:
    tipe = st.radio('Pilih Tipe', ['single vector', 'double vector', 'single matrix', 'double matrix', 'Eigen', 'OBE', 'SVD', 'Quadratic Matrix'])

with st.expander('Pilih Ukuran'):
    with st.form('Pilih Ukuran'):
        if tipe == 'single vector':
            size = st.number_input('ukuran vektor', min_value=2)
        elif tipe == 'double matrix':
            row1 = st.number_input('ukuran baris matrix pertama', min_value=2)
            col1 = st.number_input('ukuran kolom matrix pertama', min_value=2)
            row2 = st.number_input('ukuran baris matrix kedua', min_value=2)
            col2 = st.number_input('ukuran kolom matrix kedua', min_value=2)
        elif tipe == 'double vector':
            size = st.number_input('ukuran double vector', min_value=2)
        submit = st.form_submit_button('submit size')

if tipe == 'single vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 2), dtype=float)
    st.write('Masukkan data untuk vektor')
    df_input = st.experimental_data_editor(df, use_container_width=True)

elif tipe == 'double vector':
    df = pd.DataFrame(columns=range(1, size + 1), index=range(1, 3), dtype=float)
    st.write('Masukkan data untuk double vector')
    df_input = st.experimental_data_editor(df, use_container_width=True)

elif tipe == 'single matrix':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)
    st.write('Matrix:')
    st.write(df_input)
    # Operasi atau manipulasi pada matrix
    # Tambahkan kode sesuai dengan kebutuhan Anda
    Operasi = st.radio('Pilih Operasi', ['A*B', 'A+B', 'Determinan', 'Invers', 'Turunan'])
    matrix1 = df_input.fillna(0).to_numpy()
    if Operasi == 'A*B':
        # Lakukan operasi perkalian matriks dengan dirinya sendiri
        result = np.matmul(matrix1, matrix1)
        st.write(result)

    elif Operasi == 'A+B':
        # Lakukan operasi penjumlahan matriks dengan dirinya sendiri
        result = matrix1 + matrix1
        st.write(result)

    elif Operasi == 'Determinan':
        # Cari determinan matriks
        determinant = np.linalg.det(matrix1)
        st.write('Determinan:')
        st.write(determinant)

    elif Operasi == 'Turunan':
        try:
            derivative = np.gradient(matrix1)
            st.write('Matrix Derivative:')
            st.write(derivative)
        except ValueError:
            st.write('Operasi turunan tidak dapat dilakukan pada matrix ini.')

elif tipe == 'double matrix':
    df1 = pd.DataFrame(columns=range(1, col1 + 1), index=range(1, row1 + 1), dtype=float)
    st.write('Masukkan data untuk matrix pertama')
    df1_input = st.experimental_data_editor(df1, use_container_width=True, key=1)

    df2 = pd.DataFrame(columns=range(1, col2 + 1), index=range(1, row2 + 1), dtype=float)
    st.write('Masukkan data untuk matrix kedua')
    df2_input = st.experimental_data_editor(df2, use_container_width=True, key=2)

    Operasi = None  # Assign a default value
    Operasi = st.radio('Pilih Operasi', ['A*B', 'A+B', 'Determinan', 'Invers'])
    matrix1 = df1_input.fillna(0).to_numpy()
    matrix2 = df2_input.fillna(0).to_numpy()

    if Operasi == 'A*B':
        result = np.matmul(matrix1, matrix2)
        st.write(result)

    elif Operasi == 'A+B':
        result = matrix1 + matrix2
        st.write(result)

    elif Operasi == 'Determinan':
        determinant1 = np.linalg.det(matrix1)
        determinant2 = np.linalg.det(matrix2)
        st.write('Determinan Matrix Pertama:')
        st.write(determinant1)
        st.write('Determinan Matrix Kedua:')
        st.write(determinant2)

elif tipe == 'SVD':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)

    if st.button('Decompose'):
        matrix = df_input.fillna(0).to_numpy()
        U, S, V = svd(matrix)
        st.write('Matriks U:')
        st.write(U)
        st.write('Nilai Singular (S):')
        st.write(S)
        st.write('Matriks V:')
        st.write(V)

elif tipe == 'Eigen':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)

    if st.button('Decompose'):
        matrix = df_input.fillna(0).to_numpy()
        eigenvalues, eigenvectors = eig(matrix)
        st.write('Eigenvalues:')
        st.write(eigenvalues)
        st.write('Eigenvectors:')
        st.write(eigenvectors)

elif tipe == 'OBE':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)

    if st.button('Perform OBE'):
        matrix = df_input.fillna(0).to_numpy()
        augmented_matrix = np.hstack((matrix, np.eye(matrix.shape[0])))
        row_operations = []
        st.write('Augmented Matrix:')
        st.write(augmented_matrix)

        st.write('Performing OBE...')
        for i in range(matrix.shape[0] - 1):
            for j in range(i + 1, matrix.shape[0]):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
                    row_operations.append(f'R{j+1} -= ({ratio}) * R{i+1}')
                st.write(augmented_matrix)

        st.write('Row Operations:')
        for operation in row_operations:
            st.write(operation)

        st.write('Final Row-Echelon Form:')
        st.write(augmented_matrix)

elif tipe == 'Quadratic Matrix':
    row = st.number_input('ukuran baris matrix', min_value=2)
    col = st.number_input('ukuran kolom matrix', min_value=2)
    df = pd.DataFrame(columns=range(1, col + 1), index=range(1, row + 1), dtype=float)
    st.write('Masukkan data untuk matrix')
    df_input = st.experimental_data_editor(df, use_container_width=True)

    if st.button('Perform Quadratic Matrix Operations'):
        matrix = df_input.fillna(0).to_numpy()
        result = np.matmul(matrix, np.transpose(matrix))
        st.write('Matrix:')
        st.write(matrix)
        st.write('Quadratic Matrix Result:')
        st.write(result)
