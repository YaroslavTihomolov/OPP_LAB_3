#include <cmath>
#include "mpi.h"
#include <iostream>
#include <vector>

#define RANK_ROOT 0

constexpr auto n1 = 9;
constexpr auto n2 = 4;
constexpr auto n3 = 4;


void fillMatrix(std::vector<int> &A, std::vector<int> &B) {
    A.resize(n1 * n2);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            A[j + i * n2] = j + i * n2 + 1;
        }
    }
    B.resize(n2 * n3);
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; j++) {
            B[j + i * n3] = j + i * n3 + 1;
        }
    }
 }


 void gathervRoutine(std::vector<int> &resultMatrix, std::vector<int> &recvcounts, std::vector<int> &displs, int size,
                     int dims0, int dims1, int tmpMatrixColumn) {
     resultMatrix.resize(n1 * n3);
     recvcounts.resize(size);
     std::fill(recvcounts.begin(), recvcounts.end(), 1);
     displs.resize(size);
     for (int i = 0; i < dims0; i++) {
         for (int j = 0; j < dims1; j++) {
             displs[j + i * dims1] = (j * tmpMatrixColumn + i * n3 * n1 / dims0) / tmpMatrixColumn;
         }
     }
 }


 void matrixMultiply(std::vector<int> &partA, std::vector<int> &partB, std::vector<int> &multiplyRes,
                     int matrixALines, int matrixBColumns) {
     for (int i = 0; i < matrixALines; i++) {
         for (int j = 0; j < matrixBColumns; j++) {
             int sum = 0;
             for (int k = 0; k < n2; k++) {
                 sum += partA[i * n2 + k] * partB[k * matrixBColumns + j];
             }
             multiplyRes[i * matrixBColumns + j] = sum;
         }
     }
 }


 void printMatrix(std::vector<int> &matrix, int columns, int lines) {
     for (int i = 0; i < columns; i++) {
         for (int j = 0; j < lines; j++) {
             std::cout << matrix[i * n3 + j] << " ";
         }
         std::cout << '\n';
     }
 }


 void freeMpiTypes(MPI_Datatype &row_type, MPI_Datatype &col_type, MPI_Datatype &col_type_resized,
                    MPI_Datatype &matrix_back, MPI_Datatype &matrix_back_resized) {
     MPI_Type_free(&row_type);
     MPI_Type_free(&col_type);
     MPI_Type_free(&col_type_resized);
     MPI_Type_free(&matrix_back);
     MPI_Type_free(&matrix_back_resized);
}


void regTypes(MPI_Datatype &row_type, MPI_Datatype &col_type, MPI_Datatype &col_type_resized,
              MPI_Datatype &matrix_back, MPI_Datatype &matrix_back_resized, int tmpMatrixColumn, int tmpMatrixLines) {
    MPI_Type_contiguous(n2, MPI_INT, &row_type);
    MPI_Type_vector(n2, (int) tmpMatrixColumn, n3, MPI_INT, &col_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);
    MPI_Type_create_resized(col_type, 0, (int) tmpMatrixColumn * sizeof(int), &col_type_resized);
    MPI_Type_commit(&col_type_resized);
    MPI_Type_vector(tmpMatrixLines, (int) tmpMatrixColumn, n3, MPI_INT, &matrix_back);
    MPI_Type_commit(&matrix_back);
    MPI_Type_create_resized(matrix_back, 0, (int) tmpMatrixColumn * sizeof(int), &matrix_back_resized);
    MPI_Type_commit(&matrix_back_resized);
}


void regComm2D(int size, int dims[2], int periods[2], MPI_Comm &comm2d, int reorder, int coords[2]) {
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
}


void Run(int size, int rank) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 0;

    MPI_Comm comm2d;

    regComm2D(size, dims, periods, comm2d, reorder, coords);

    int tmpMatrixColumn = n3 / dims[1];
    int tmpMatrixLines = n1 / dims[0];

    MPI_Datatype row_type, col_type, col_type_resized, matrix_back, matrix_back_resized;
    regTypes(row_type, col_type, col_type_resized, matrix_back, matrix_back_resized, tmpMatrixColumn, tmpMatrixLines);

    MPI_Comm commX, commY;
    MPI_Comm_split(comm2d, coords[0], rank, &commX);
    MPI_Comm_split(comm2d, coords[1], rank, &commY);

    const auto p1 = n1 * n2 / dims[0];
    const auto p2 = n2 * tmpMatrixColumn;

    std::vector<int> partA(p1);
    std::vector<int> partB(p2);

    std::vector<int> A, B;

    if (rank == RANK_ROOT) {
        fillMatrix(A, B);
    }

    if (coords[1] == 0) {
        MPI_Scatter(A.data(), tmpMatrixLines, row_type, partA.data(), tmpMatrixLines, row_type, RANK_ROOT, commY);
    }

    if (coords[0] == 0) {
        MPI_Scatter(B.data(), 1, col_type_resized, partB.data(), tmpMatrixColumn * n2, MPI_INT, RANK_ROOT, commX);
    }


    MPI_Bcast(partA.data(), p1, MPI_INT, RANK_ROOT, commX);
    MPI_Bcast(partB.data(), p2, MPI_INT, RANK_ROOT, commY);

    std::vector<int> multiplyRes(tmpMatrixLines * tmpMatrixColumn);

    matrixMultiply(partA, partB, multiplyRes, tmpMatrixLines, tmpMatrixColumn);

    std::vector<int> resultMatrix, recvcounts, displs;
    if (rank == RANK_ROOT) {
        gathervRoutine(resultMatrix, recvcounts, displs, size, dims[0], dims[1], tmpMatrixColumn);
    }

    MPI_Gatherv(multiplyRes.data(), multiplyRes.size(), MPI_INT, resultMatrix.data(), recvcounts.data(), displs.data(),
                matrix_back_resized, RANK_ROOT, MPI_COMM_WORLD);

    if (rank == RANK_ROOT) {
        printMatrix(resultMatrix, n1, n3);
    }

    freeMpiTypes(row_type, col_type, col_type_resized, matrix_back, matrix_back_resized);
}


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Run(size, rank);
    MPI_Finalize();
    return 0;
}
