// Minimal check: can a nanobind-wrapped C function do real MPI work using the
// communicator handed in from mpi4py? Each rank sums its slice of the midpoint rule for
// pi, then MPI_Allreduce combines the partial sums. If the ranks did NOT message each
// other, rank 0 would only see its own slice and the result would not be pi.
#include <nanobind/nanobind.h>
#include <mpi.h>

namespace nb = nanobind;
using namespace nb::literals;

// comm_f is mpi4py's Fortran handle (comm.py2f()); MPI_Comm_f2c turns it back into the
// C communicator, which is the portable way to pass a communicator across the boundary.
static double pi_mpi(int comm_f, long n)
{
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const double h = 1.0 / (double)n;
    const long lo = n * (long)rank / size;
    const long hi = n * (long)(rank + 1) / size;
    double local = 0.0;
    for (long i = lo; i < hi; i++) {
        const double x = (i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global * h;
}

static nb::tuple rank_size(int comm_f)
{
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    return nb::make_tuple(rank, size);
}

NB_MODULE(pi_mpi_ext, m)
{
    m.def("pi_mpi", &pi_mpi, "comm_f"_a, "n"_a,
          "Distributed midpoint-rule pi over the given mpi4py communicator handle.");
    m.def("rank_size", &rank_size, "comm_f"_a,
          "Return (rank, size) for the given mpi4py communicator handle.");
}
