#include <cmath>
#include "mpi.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <unistd.h>

#define RANK_ROOT 0

constexpr auto n1 = 4;
constexpr auto n2 = 4;
constexpr auto n3 = 4;


/*typedef struct {
    int *data;
    int cols;
    int rows;
} Mat;


void mat_fill(
        Mat mat,
        int comm_size) {
    assert(mat.cols % comm_size == 0);

    const int columns_per_process = mat.cols / comm_size;

    for (int y = 0; y < mat.rows; y += 1) {
        for (int proc_index = 0; proc_index < comm_size; proc_index += 1) {
            for (int x = columns_per_process * proc_index; x < columns_per_process * (proc_index + 1); x += 1) {
                mat.data[y * mat.cols + x] = proc_index;
            }
        }
    }
}


void mat_print(Mat mat) {
    for (int i = 0; i < mat.rows * mat.cols; i += 1) {
        printf("%d ", mat.data[i]);

        if ((i + 1) % mat.cols == 0) {
            printf("\n");
        }
    }
}


int run(int size, int rank) {
    const int N = 8;
    assert(N % size == 0);

    const int columns_per_process = N / size;

    int *data0 = nullptr;
    int *data;

    if (RANK_ROOT == rank) {
        data0 = (int *) calloc(N * N, sizeof(*data));
        mat_fill((Mat) {data0, N, N}, size);
        printf("ROOT:\n");
        mat_print((Mat) {data0, N, N});
    }

    data = (int *) calloc(N * columns_per_process, sizeof(*data));


    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Type_vector(N, columns_per_process, N, MPI_INT, &vertical_int_slice);
    MPI_Type_commit(&vertical_int_slice);

    MPI_Type_create_resized(vertical_int_slice, 0, (int) (columns_per_process * sizeof(*data)), &vertical_int_slice_resized);

    MPI_Type_commit(&vertical_int_slice_resized);

    MPI_Scatter(data0, 1, vertical_int_slice_resized, data, N * columns_per_process, MPI_INT, RANK_ROOT, MPI_COMM_WORLD);

    printf("RANK %d:\n", rank);
    mat_print((Mat) {data, .rows = N, .cols = columns_per_process});

    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    free(data);
    free(data0);
    return EXIT_SUCCESS;
}*/


int main(int argc, char **argv) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 0;

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm comm2d;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(size, 2, dims);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    MPI_Datatype row_type, col_type, col_type_resized;
    MPI_Type_contiguous(n2, MPI_INT, &row_type);
    MPI_Type_vector(n2, (int) n3 / dims[1], n3, MPI_INT, &col_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);
    MPI_Type_create_resized(col_type, 0, (int) n3 / dims[1] * sizeof(int), &col_type_resized);
    MPI_Type_commit(&col_type_resized);

    MPI_Comm commX, commY;

    MPI_Comm_split(comm2d, coords[0], rank, &commX);
    MPI_Comm_split(comm2d, coords[1], rank, &commY);

    const auto p1 = n1 * n2 / dims[0];
    const auto p2 = n2 * n3 / dims[1];

    std::vector<int> partA(p1);
    std::vector<int> partB(p2);


    std::vector<int> A;
    std::vector<int> B;
    if (rank == RANK_ROOT) {
        A.resize(n1 * n2);
        int tmp = 0;
        for (auto i = 0; i < n1 * n2; i++) {
            A[i] = tmp++;
        }
        B.resize(n2 * n3);
        for (auto i = 0; i < n2 * n3; i++) {
            B[i] = tmp++;
        }
    }
    if (coords[1] == 0) {
        MPI_Scatter(A.data(), n1 / dims[0], row_type, partA.data(), n1 / dims[0], row_type, RANK_ROOT, commY);
    }
    if (coords[0] == 0) {
        MPI_Scatter(B.data(), 1, col_type_resized, partB.data(), n3 / dims[1] * n2, MPI_INT, RANK_ROOT, commX);
    }


    MPI_Bcast(partA.data(), p1, MPI_INT, RANK_ROOT, commX);
    MPI_Bcast(partB.data(), p2, MPI_INT, RANK_ROOT, commY);


    sleep(rank);
    std::cout << rank << ": ";
    for (auto i: partA) {
        std::cout << i << " ";
    }
    for (auto i: partB) {
        std::cout << i << " ";
    }
    std::cout << '\n';

    MPI_Finalize();
    return 0;
}
