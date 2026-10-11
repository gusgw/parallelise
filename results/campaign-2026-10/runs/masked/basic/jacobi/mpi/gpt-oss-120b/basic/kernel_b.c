        /* Halo exchange with neighbours */
        int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        MPI_Status status;
        /* Send first interior row upward (tag 0), receive top halo from up (tag 1) */
        MPI_Sendrecv(&cur[1 * n], n, MPI_DOUBLE, up, 0,
                     &cur[0 * n], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, &status);
        /* Send last interior row downward (tag 1), receive bottom halo from down (tag 0) */
        MPI_Sendrecv(&cur[local_n * n], n, MPI_DOUBLE, down, 1,
                     &cur[(local_n + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, &status);
