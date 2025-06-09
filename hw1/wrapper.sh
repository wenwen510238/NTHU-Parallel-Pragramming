#! /bin/bash

mkdir -p nsys_reports_2

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
-o "./nsys_reports_2/rank_pointer_$PMI_RANK.nsys-rep" \
--mpi-impl openmpi \
--trace mpi,ucx,osrt,nvtx \
--force-overwrite true \
$@

