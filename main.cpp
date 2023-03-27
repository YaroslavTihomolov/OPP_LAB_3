#include <cmath>
#include "mpi.h"
#include <iostream>
#include <vector>


int main(int argc, char **argv) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 0;
    int size, rank, sizex, sizey, ranky, rankx;
    int prevy, prevx, nexty, nextx;
    MPI_Init(&argc, &argv);
    MPI_Comm comm2d;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(size, 2, dims);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Cart_get(comm2d,2,dims,periods,coords);
    sizey = dims[0];
    sizex = dims[1];

    ranky = coords[0];
    rankx = coords[1];

    MPI_Comm commX, commY;

    MPI_Comm_split(comm2d, coords[0], rank, &commY);
    MPI_Comm_split(comm2d, coords[1], rank, &commX);
    int size1;
    MPI_Comm_rank(commY, &size1);
    printf("%d: %d\n", rank, size1);



    MPI_Finalize();
    return 0;
}
