# Contributing

## Setting up

You need Python 3.11 or later, `gcc`, `make` and, for the MPI (Message Passing Interface) tests,
Open MPI. Clone the repository, make a virtual environment, and install the package with its
development tools:

    python3 -m venv .venv && . .venv/bin/activate
    make install

## Running the checks

    make test    # runs every test; the CUDA tests skip themselves without a graphics card
    make lint    # runs ruff (style) and mypy (types)

Both must pass before a change is offered. `make test-fast` skips the tests that compile C code
and is useful while you work.

## Generated code is untrusted

The program compiles and runs C code that a language model wrote. Treat that code as hostile.
Never run it outside the harness, never give it a secret, and do not widen the list of commands
the agent may run without a clear reason. The allow-list and the secret-free environment limit
the damage but are not a sandbox; a container is the real boundary. The section on security in
the README explains this.

## Writing

Write documents, code comments and commit messages in plain, clear, concise, correct English,
in full sentences with ordinary words. Spell out each abbreviation the first time you use it.
A document describes what the code does, for a reader who is new to it. A commit message says
what the change does and why, in a sentence or two, not only which files it touches.
