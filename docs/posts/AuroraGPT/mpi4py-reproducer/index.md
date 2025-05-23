# 🐛 `mpi4py` bug on Sunspot
Sam Foreman
2024-05-25

<link rel="preconnect" href="https://fonts.googleapis.com">

Simple reproducer:

1.  Load my `anl_24_q2_release` conda environment:

    ``` bash
    #[08:42:38 AM][foremans@x1922c2s3b0n0][~]
    $ eval "$(~/miniconda3/bin/conda shell.zsh hook)" ; conda activate anl_24_q2_release
    ```

2.  Try `python3 -c 'from mpi4py import MPI'`

    - fails ❌

    ``` bash
    # [08:44:41 AM][foremans@x1922c2s3b0n0][~][anl_24_q2_release]
    $ python3 -c 'from mpi4py import MPI'
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: /home/foremans/miniconda3/envs/anl_24_q2_release/lib/python3.9/site-packages/mpi4py/MPI.cpython-39-x86_64-linux-gnu.so: undefined symbol: MPI_Message_c2f
    [1]    14910 exit 1     python3 -c 'from mpi4py import MPI'
    ```

3.  Load correct modules:

    ``` bash
    # [08:44:58 AM][foremans@x1922c2s3b0n0][~][anl_24_q2_release]
    $ module use /home/ftartagl/graphics-compute-runtime/modulefiles ; module load graphics-compute-runtime/agama-ci-devel-803.29 spack-pe-gcc/0.6.1-23.275.2 gcc/12.2.0 ; module use /soft/preview-modulefiles/24.086.0 ; module load oneapi/release/2024.04.15.001
         UMD: agama-ci-devel-803.29 successfully loaded:
         UMD: graphics-compute-runtime/agama-ci-devel-803.29

    Due to MODULEPATH changes, the following have been reloaded:
      1) mpich-config/collective-tuning/1024

    The following have been reloaded with a version change:
      1) intel_compute_runtime/release/agama-devel-736.25 => intel_compute_runtime/release/775.20     2) mpich/icc-all-pmix-gpu/52.2 => mpich/icc-all-pmix-gpu/20231026     3) oneapi/eng-compiler/2023.12.15.002 => oneapi/release/2024.04.15.001
    ```

4.  Retry with new modules:

    - works ✅

    ``` bash
    # [08:45:01 AM][foremans@x1922c2s3b0n0][~][anl_24_q2_release]
    $ python3 -c 'from mpi4py import MPI; print(MPI.__file__)'
    /home/foremans/miniconda3/envs/anl_24_q2_release/lib/python3.9/site-packages/mpi4py/MPI.cpython-39-x86_64-linux-gnu.so
    ```
