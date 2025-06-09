#! /bin/bash

mkdir -p nsys_reports_12

# # Output to ./nsys_reports/rank_$N.nsys-rep
# nsys profile \
# -o "./nsys_reports_2/pthread_slow_01_8.nsys-rep" \
# --trace osrt,nvtx \
# --force-overwrite true \
# $@

nsys profile \
-o "./nsys_reports_12/rank_pointer_$PMI_RANK.nsys-rep" \
--mpi-impl openmpi \
--trace mpi,osrt,nvtx \
--force-overwrite true \
$@
