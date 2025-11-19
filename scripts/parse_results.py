#read jobs results into one big csv

from sbatchman import jobs_list

jobs = jobs_list(from_archived=True, tag = "CuSPARSE_CSR_SPMM", status = ["COMPLETED",])
for job in jobs:
    command, pos_args, named_args = job.parse_command_args()
    print(command)
    print(pos_args)
    print(named_args)
    print(job.status)
    print(job.variables)
    print(job.get_stdout())

