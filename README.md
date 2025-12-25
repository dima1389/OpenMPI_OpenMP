# MPI Compilation and Environment Setup

This document provides a concise and technically precise guide for installing MPI libraries and compiling MPI programs in **C/C++** on **Linux** and **Windows** systems.

---

## Linux: Ubuntu (OpenMPI)

### Installation

Install **OpenMPI** and the required development headers using `apt`:

```bash
sudo apt-get install libopenmpi-dev openmpi-*
```

This installs:

* MPI runtime (`mpirun`, `mpiexec`)
* MPI compiler wrappers (`mpicc`, `mpic++`)
* Header files and libraries

---

### Compilation

Always use MPI compiler wrappers.

#### C programs

```bash
mpicc -o mpi_program mpi_program.c
```

#### C++ programs

```bash
mpic++ -o mpi_program mpi_program.cpp
```

The wrapper automatically:

* invokes `gcc` / `g++`,
* adds include paths,
* links MPI libraries.

No manual configuration is required.

---

## Windows: Microsoft MPI

### Installation

Download and install **Microsoft MPI v10.1.3** from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=105289).

Install both components:

* `msmpisetup.exe` – MPI runtime
* `msmpisdk.msi` – headers and libraries (SDK)

After installation, the following environment variables are available:

| Variable      | Description                  |
| ------------- | ---------------------------- |
| `MSMPI_INC`   | Path to MPI header files     |
| `MSMPI_LIB64` | Path to 64-bit MPI libraries |

---

### Compilation

Unlike Linux, MPI compiler wrappers are not provided for MinGW. Compilation must be done manually using `gcc` or `g++`.

#### Using MinGW64 (MSYS2)

```bash
gcc mpi_program.c -I"$MSMPI_INC" -L"$MSMPI_LIB64" -lmsmpi -o mpi_program.exe
```

#### Using Windows Command Prompt (CMD)

```cmd
g++ mpi_program.cpp -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o mpi_program.exe
```

---

## Executing an MPI Program

MPI programs can run using **multiple processes at the same time**.

If the MPI executable is started directly, it runs with **one process only**.
To run it with multiple processes, use the `mpiexec` command.

```bash
mpiexec -n <num_processes> <executable_path>
```

* `<num_processes>` – number of processes to run
* `<executable_path>` – path to the MPI executable

**Example (run with 4 processes):**

```bash
mpiexec -n 4 mpi_program
```

---

## Notes and Best Practices

* Prefer `mpicc` / `mpic++` whenever available.
* On Linux, MPI linking is fully automated.
* On Windows with MinGW, include and library paths must be specified explicitly.
* Do not mix MSVC (`cl.exe`) and MinGW (`g++`) toolchains in the same build process.
