        if (local_rows > 0) {
            /* Copy interior rows into halo buffer */
            memcpy(p_halo + m, p_loc, local_n * sizeof(double));

            int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
            int down = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
            /* Exchange with the rank above: send first row, receive neighbor's last row */
            MPI_Sendrecv(p_loc,                     /* send first local row */
                         m, MPI_DOUBLE, up, 0,
                         p_halo,                    /* receive into halo[0] */
                         m, MPI_DOUBLE, up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Exchange with the rank below: send last row, receive neighbor's first row */
            MPI_Sendrecv(p_loc + (local_rows - 1) * m, /* send last local row */
                         m, MPI_DOUBLE, down, 1,
                         p_halo + (local_rows + 1) * m,   /* receive into halo[last+1] */
                         m, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Apply A to p using 5‑point stencil */
            for (long i = 0; i < local_rows; ++i) {
                long global_i = lo + i;
                for (long j = 0; j < m; ++j) {
                    double s = 4.0 * p_halo[(i + 1) * m + j];
                    if (global_i > 0)        s -= p_halo[i * m + j];          /* north */
                    if (global_i < m - 1)    s -= p_halo[(i + 2) * m + j];      /* south */
                    if (j > 0)               s -= p_halo[(i + 1) * m + (j - 1)];/* west */
                    if (j < m - 1)           s -= p_halo[(i + 1) * m + (j + 1)];/* east */
                    q_loc[i * m + j] = s;
                }
            }
        }
