"""Generate the C files for a declared routine: its header, the driver that calls it, a Makefile.

Every backend gets the same header text and the same driver, apart from a few lines: MPI
starts its processes, shares the inputs and waits for every process, and CUDA waits for the
graphics card. The driver takes commands on its standard input, each of three lines: CALL
(call the routine once, untimed) or TIME (the same, between a START and an END mark), then the
inputs file, then the results file. It writes START, END and DONE marks, one per line, to the
named pipe given as its only argument, and after START it waits for the line GO before it calls
the routine, so that the harness notes the START mark while a processor is free. The harness
times a call by the arrival of the START and END marks, so a routine cannot forge its own
timing; and since a routine may read no file (see ``rules``), it cannot write to the pipe
either. For MPI, only process 0 reads commands and files
and writes marks, every process holds all the inputs when the routine is called, and the other
processes wait for each command asleep, so that they take no processor time from a program that
is being timed meanwhile.

The serial routine of a problem is built with the same driver, for the plain C backend
"serial", so that it is timed in exactly the same way as the routine it is compared with. The
driver's own names all begin with ``drv_``, which no declared name may do.
"""

from __future__ import annotations

from pathlib import Path

from .routine import C_TYPES, Declaration, Value, c_prototype

BACKENDS = ("openmp", "mpi", "cuda")
MPI_TYPES = {
    "float64": "MPI_DOUBLE", "float32": "MPI_FLOAT", "int64": "MPI_LONG", "int32": "MPI_INT",
}

CONTRACT = """\
/* {name}: generated from a notebook's declaration by parallelise. Do not change this file.
 *
 * Write the routine declared below so that it gives the same results as the notebook's
 * function for any arguments of the declared shapes. Every process receives all the
 * arguments. When the routine returns, every result must be complete in the caller's
 * memory: for MPI, on process 0; for CUDA, in ordinary memory, copied back from the
 * graphics card. The routine must compute its results from its arguments alone: it may keep
 * nothing from one call to the next, read no file, no clock and no environment variable,
 * and start no process or thread except through its backend. It may use the C library's
 * mathematics, memory and printing functions and the calls of its backend.
 */
#ifndef {guard}
#define {guard}

#ifdef __cplusplus
extern "C" {{
#endif

{prototype};

#ifdef __cplusplus
}}
#endif

#endif
"""


def header(declaration: Declaration, name: str) -> str:
    """Return the routine's header file."""
    return CONTRACT.format(
        name=name, guard=f"{name.upper()}_H", prototype=c_prototype(declaration, name)
    )


def routine_file(name: str, backend: str) -> str:
    """Return the name of the file in which the routine is written."""
    return f"{name}.cu" if backend == "cuda" else f"{name}.c"


# --- the driver ----------------------------------------------------------------------------------


def _count(value: Value) -> str:
    return " * ".join(f"(size_t){s}" for s in value.shape)


def _read(declaration: Declaration) -> list[str]:
    """The lines that read and share the sizes and the arguments, and prepare the results."""
    sizes = declaration.sizes
    lines = [f"    long drv_sizes[{max(len(sizes), 1)}] = {{0}};", "    (void)drv_sizes;"]
    if sizes:
        lines.append(f"    drv_load(drv_in, drv_sizes, sizeof(long) * {len(sizes)});")
        lines.append(f"    drv_share(drv_sizes, {len(sizes)}, MPI_LONG);")
    lines += [f"    long {s} = drv_sizes[{i}];\n    (void){s};" for i, s in enumerate(sizes)]
    for a in declaration.arguments:
        ctype, mpi = C_TYPES[a.dtype], MPI_TYPES[a.dtype]
        if a.name in sizes:  # an integer argument that is also a size: use the size
            lines.append(f"    drv_skip(drv_in, sizeof({ctype}));")
        elif a.is_array:
            bytes_ = f"sizeof({ctype}) * {_count(a)}"
            lines.append(f"    {ctype} *{a.name} = ({ctype} *)drv_take({bytes_});")
            lines.append(f"    drv_load(drv_in, {a.name}, {bytes_});")
            lines.append(f"    drv_share({a.name}, {_count(a)}, {mpi});")
        else:
            lines.append(f"    {ctype} {a.name} = 0;")
            lines.append(f"    drv_load(drv_in, &{a.name}, sizeof {a.name});")
            lines.append(f"    drv_share(&{a.name}, 1, {mpi});")
    for r in declaration.results:
        if r.name in declaration.updated or declaration.returns_value():
            continue
        ctype = C_TYPES[r.dtype]
        if r.is_array:
            bytes_ = f"sizeof({ctype}) * {_count(r)}"
            lines.append(f"    {ctype} *{r.name} = ({ctype} *)drv_take({bytes_});")
            lines.append(f"    drv_spoil({r.name}, {bytes_});")
        else:
            lines.append(f"    {ctype} {r.name};")
            lines.append(f"    drv_spoil(&{r.name}, sizeof {r.name});")
        if r.bounded:
            lines.append(f"    long {r.name}_count = -1;")
    return lines


def _call(declaration: Declaration, name: str) -> str:
    """The line that calls the routine."""
    arguments = {a.name: a for a in declaration.arguments}
    results = {r.name: r for r in declaration.results}
    parameters = []
    for item in declaration.c_order:
        if item in arguments and item in declaration.sizes:
            parameters.append(f"({C_TYPES[arguments[item].dtype]}){item}")
        elif item in results and item not in arguments:
            parameters.append(item if results[item].is_array else f"&{item}")
            if results[item].bounded:
                parameters.append(f"&{item}_count")
        else:
            parameters.append(item)
    call = f"{name}({', '.join(parameters)})"
    if declaration.returns_value():
        result = declaration.results[0]
        return f"        {C_TYPES[result.dtype]} {result.name} = {call};"
    return f"        {call};"


def _write(declaration: Declaration) -> list[str]:
    """The lines that write the results."""
    lines = []
    for r in declaration.results:
        ctype = C_TYPES[r.dtype]
        if r.bounded:
            lines.append(
                f"        if (drv_rank == 0 && ({r.name}_count < 0 || "
                f"(size_t){r.name}_count > {_count(r)}))\n"
                f"            drv_fail(\"the length of {r.name}\");"
            )
            lines.append(f"        drv_save(drv_out, &{r.name}_count, sizeof(long));")
            length = f"sizeof({ctype}) * (size_t){r.name}_count"
            lines.append(f"        drv_save(drv_out, {r.name}, {length});")
        elif r.is_array:
            lines.append(f"        drv_save(drv_out, {r.name}, sizeof({ctype}) * {_count(r)});")
        else:
            lines.append(f"        drv_save(drv_out, &{r.name}, sizeof {r.name});")
    return lines


def _free(declaration: Declaration) -> list[str]:
    arrays = [a.name for a in declaration.arguments if a.is_array]
    arrays += [
        r.name for r in declaration.results if r.is_array and r.name not in declaration.updated
    ]
    return [f"    free({n});" for n in arrays]


DRIVER = """\
/* The driver for {name} on {backend}, generated by parallelise. Do not change this file. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
{includes}
#include "{name}.h"

static int drv_rank = 0;

static inline void drv_fail(const char *what)
{{
    fprintf(stderr, "driver: cannot handle %s\\n", what);
    exit(2);
}}

static inline void *drv_take(size_t bytes)
{{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) drv_fail("the memory it needs");
    return p;
}}

/* Only process 0 reads and writes files; the others receive what it read. */
static inline void drv_load(FILE *f, void *p, size_t bytes)
{{
    if (drv_rank == 0 && bytes && fread(p, 1, bytes, f) != bytes) drv_fail("the inputs");
}}

static inline void drv_skip(FILE *f, size_t bytes)
{{
    if (drv_rank == 0 && fseek(f, (long)bytes, SEEK_CUR) != 0) drv_fail("the inputs");
}}

static inline void drv_save(FILE *f, const void *p, size_t bytes)
{{
    if (drv_rank == 0 && bytes && fwrite(p, 1, bytes, f) != bytes) drv_fail("the results");
}}

/* Fill a result, before each call, with bytes that no routine could leave by accident. */
static inline void drv_spoil(void *p, size_t bytes)
{{
    unsigned char *c = (unsigned char *)p;
    for (size_t i = 0; i < bytes; i++) c[i] = (unsigned char)(rand() | 0x80);
}}
{share}
static inline void drv_mark(FILE *marks, const char *what)
{{
    if (drv_rank == 0) {{
        fputs(what, marks);
        fputc('\\n', marks);
        fflush(marks);
    }}
}}

/* Read one line of a command, without its newline: a command gives each word on a line. */
static inline int drv_line(char *line, int size)
{{
    if (!fgets(line, size, stdin)) return 0;
    line[strcspn(line, "\\n")] = 0;
    return 1;
}}

static void drv_call(FILE *drv_in, FILE *drv_out, int drv_timed, FILE *drv_marks)
{{
{read}
    {{
        {before}
        if (drv_timed) {{
            drv_mark(drv_marks, "START");
            drv_go();
        }}
{call}
        {after}
        if (drv_timed) drv_mark(drv_marks, "END");
{write}
    }}
{free}
}}

int main(int argc, char **argv)
{{
{start}
    if (argc != 2) {{
        fprintf(stderr, "usage: driver MARKS-PIPE\\n");
        return 2;
    }}
    FILE *marks = drv_rank == 0 ? fopen(argv[1], "w") : NULL;
    if (drv_rank == 0 && !marks) drv_fail("the marks pipe");
    char command[16], inputs[4096], results[4096];
    for (;;) {{
        int order[2] = {{0, 0}}; /* whether a command came, and whether it is to be timed */
        if (drv_rank == 0) {{
            order[0] = drv_line(command, sizeof command) && drv_line(inputs, sizeof inputs)
                       && drv_line(results, sizeof results);
            order[1] = order[0] && strcmp(command, "TIME") == 0;
        }}
        drv_wait(order, 2);
        if (!order[0]) break;
        int timed = order[1];
        FILE *in = drv_rank == 0 ? fopen(inputs, "rb") : NULL;
        FILE *out = drv_rank == 0 ? fopen(results, "wb") : NULL;
        if (drv_rank == 0 && (!in || !out)) drv_fail("a file named in a command");
        drv_call(in, out, timed, marks);
        if (drv_rank == 0) {{
            fclose(in);
            fclose(out);
        }}
        drv_mark(marks, "DONE");
    }}
{finish}
    return 0;
}}
"""

_SHARE_MPI = """
static inline void drv_share(void *p, size_t count, MPI_Datatype type)
{
    int size = 0;
    MPI_Type_size(type, &size);
    for (size_t done = 0; done < count;) {
        size_t piece = count - done < ((size_t)1 << 28) ? count - done : ((size_t)1 << 28);
        MPI_Bcast((char *)p + done * (size_t)size, (int)piece, type, 0, MPI_COMM_WORLD);
        done += piece;
    }
}

/* Wait until the harness has noted the START mark before the routine starts, so that it reads
 * the mark while a processor is free and the routine's own work cannot delay it. */
static inline void drv_go(void)
{
    char line[8];
    if (drv_rank == 0 && !fgets(line, sizeof line, stdin)) drv_fail("the signal to start");
    MPI_Barrier(MPI_COMM_WORLD);
}

/* The other processes wait for each command asleep, so that they take no processor time from
 * a program that is being timed meanwhile, such as the serial routine. */
static inline void drv_wait(int *values, int count)
{
    MPI_Request request;
    int done = 0;
    MPI_Ibcast(values, count, MPI_INT, 0, MPI_COMM_WORLD, &request);
    while (MPI_Test(&request, &done, MPI_STATUS_IGNORE) == MPI_SUCCESS && !done) {
        struct timespec pause = {0, 1000000};
        nanosleep(&pause, NULL);
    }
}
"""

_SHARE_NONE = """
#define MPI_LONG 0
#define MPI_INT 0
#define MPI_DOUBLE 0
#define MPI_FLOAT 0
static inline void drv_share(void *p, size_t count, int type)
{
    (void)p;
    (void)count;
    (void)type;
}

static inline void drv_wait(int *values, int count)
{
    (void)values;
    (void)count;
}

/* Wait until the harness has noted the START mark before the routine starts, so that it reads
 * the mark while a processor is free and the routine's own work cannot delay it. */
static inline void drv_go(void)
{
    char line[8];
    if (!fgets(line, sizeof line, stdin)) drv_fail("the signal to start");
}
"""


def driver(declaration: Declaration, name: str, backend: str) -> str:
    """Return the driver's source for a backend: openmp, mpi, cuda or serial."""
    mpi, cuda = backend == "mpi", backend == "cuda"
    after = ""
    if mpi:
        after = "MPI_Barrier(MPI_COMM_WORLD);"
    elif cuda:
        after = 'if (cudaDeviceSynchronize() != cudaSuccess) drv_fail("the graphics card");'
    return DRIVER.format(
        name=name,
        backend=backend,
        includes="#include <mpi.h>\n#include <time.h>" if mpi
        else "#include <cuda_runtime.h>" if cuda else "",
        share=_SHARE_MPI if mpi else _SHARE_NONE,
        read="\n".join(_read(declaration)),
        before="MPI_Barrier(MPI_COMM_WORLD);" if mpi else "",
        call=_call(declaration, name),
        after=after,
        write="\n".join(_write(declaration)),
        free="\n".join(_free(declaration)),
        start="    MPI_Init(&argc, &argv);\n    MPI_Comm_rank(MPI_COMM_WORLD, &drv_rank);"
        if mpi else "",
        finish="    MPI_Finalize();" if mpi else "",
    )


# --- the Makefile --------------------------------------------------------------------------------

MAKEFILE = """\
# Builds {name} for {backend}, and the problem's serial routine, each with the generated driver.
# Generated by parallelise. Do not change this file.
CC = {cc}
FLAGS = {flags}
SERIAL_FLAGS = -O2 -Wall -Wextra -Werror -std=c11

all: driver serial_driver

routine.o: {source} {name}.h
\t$(CC) $(FLAGS) -c {source} -o routine.o

driver: {driver} routine.o
\t$(CC) $(FLAGS) {driver} routine.o -o driver -lm

serial.o: serial.c {name}.h
\tgcc $(SERIAL_FLAGS) -c serial.c -o serial.o

serial_driver: serial_driver.c serial.o
\tgcc $(SERIAL_FLAGS) serial_driver.c serial.o -o serial_driver -lm

.PHONY: all
"""

COMPILERS = {
    "serial": ("gcc", "-O2 -Wall -Wextra -Werror -std=c11"),
    "openmp": ("gcc", "-O2 -fopenmp -Wall -Wextra -Werror -std=c11"),
    "mpi": ("mpicc", "-O2 -Wall -Wextra -Werror -std=c11"),
    "cuda": ("nvcc", "-O2 -Xcompiler -Wall,-Wextra,-Werror"),
}


def makefile(name: str, backend: str) -> str:
    cc, flags = COMPILERS[backend]
    return MAKEFILE.format(
        name=name, backend=backend, cc=cc, flags=flags, source=routine_file(name, backend),
        driver="driver.cu" if backend == "cuda" else "driver.c",
    )


def write_build_files(folder: Path, declaration: Declaration, name: str, backend: str) -> None:
    """Write the header, both drivers and the Makefile into a folder."""
    (folder / f"{name}.h").write_text(header(declaration, name), encoding="utf-8")
    driver_name = "driver.cu" if backend == "cuda" else "driver.c"
    (folder / driver_name).write_text(driver(declaration, name, backend), encoding="utf-8")
    (folder / "serial_driver.c").write_text(driver(declaration, name, "serial"), encoding="utf-8")
    (folder / "Makefile").write_text(makefile(name, backend), encoding="utf-8")
