# Two build stages. The "cpu" stage is the default and is what the continuous
# integration workflow builds. The "cuda" stage starts from an NVIDIA development image so
# that the CUDA backend can be compiled and run on a machine with a graphics card.
#
#   docker build --target cpu  -t parallelise:cpu  .
#   docker build --target cuda -t parallelise:cuda .

FROM python:3.12-slim AS cpu

# The compilers and the Message Passing Interface (MPI) implementation that the
# generated C kernels are built with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc make openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# The programs run generated code, so nothing here runs as the root user.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin parallelise

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[dev,report,agents]" \
    && chown -R parallelise:parallelise /app

USER parallelise
ENV PARALLELISE_WORK_ROOT=/work \
    PARALLELISE_RESULTS=/app/results/runs.jsonl
# Open MPI refuses to start more processes than it sees cores unless told otherwise;
# oversubscription lets the many-worker tests run on a small container.
ENV OMPI_MCA_rmaps_base_oversubscribe=1
ENTRYPOINT ["parallelise"]
CMD ["--help"]


FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv gcc make openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships a Python that refuses system-wide installs, so use a virtual
# environment and put it first on the search path.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The base image already defines a user with identifier 1000 on this release, so choose
# a different one.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin parallelise

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[dev,report,agents]" \
    && chown -R parallelise:parallelise /app

USER parallelise
ENV PARALLELISE_WORK_ROOT=/work \
    PARALLELISE_RESULTS=/app/results/runs.jsonl \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    NVCC=/usr/local/cuda/bin/nvcc
# The CUDA Makefiles default to the compiler path used on the author's workstation; the
# line above points them at the compiler in this image instead.
ENTRYPOINT ["parallelise"]
CMD ["--help"]
