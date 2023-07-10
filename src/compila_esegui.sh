#module avail
#module load
#module list

clear

#Run your job
sbatch job.sh
#Check your job
#squeue –j <your job id>
#Check your job by user
squeue -U $USER
#View partition and node information
#sinfo
#Cancel your job
#scancel <jobId>

cat output/err.txt
cat output/out.txt