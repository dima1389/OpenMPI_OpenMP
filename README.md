# Microsoft MPI

Download and install Microsoft MPI v10.1.3 from the official [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=105289):
- `msmpisetup.exe`
- `msmpisdk.msi`

# Ubuntu

- `apt-get install libopenmpi-dev openmpi-*`

## MINGW64

```bash
Dimitrije@DESKTOP-EO2KI6T MINGW64 /d/Projects/MPS
$ g++ MPI_Hello_World.c -I"$MSMPI_INC" -L"$MSMPI_LIB64" -lmsmpi -o MPI_Hello_World.exe
```

## CMD

```bash
D:\Projects\MPS>g++ MPI_Hello_World.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Hello_World.exe
```
