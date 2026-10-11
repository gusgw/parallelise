        /* Exchange halo rows of p with neighbours */
        if (local_rows > 0) {
            int up   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
            int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
            /* Send first local row upward, receive row from above (send tag 0, recv tag 1) */
            MPI_Sendrecv(p_loc,                m, MPI_DOUBLE, up,   0,
                         halo_up_recv,        m, MPI_DOUBLE, up,   1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Send last local row downward, receive row from below (send tag 1, recv tag 0) */
            MPI_Sendrecv(p_loc + (local_rows - 1) * m, m, MPI_DOUBLE, down, 1,
                         halo_down_recv,      m, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
