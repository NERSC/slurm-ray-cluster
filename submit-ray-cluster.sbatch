#!/bin/bash

#SBATCH -C gpu
#SBATCH --time=00:10:00
#SBATCH -q debug
#SBATCH -A nstaff

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128


# Load modules or your own conda environment here
module load pytorch/1.10.0

# Set a unique 'password' for the edge case where two users of a shared node are both using ray
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(hostname)
head_node=$(hostname)
port=6379
ip_head=$head_node:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w $head_node start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
#for ((  i=1; i<=$worker_num; i++ ))
#do
#  node_i=${nodes_array[$i]}
#  echo "STARTING WORKER $i at $node_i"
#  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
#  sleep 5
#done

echo srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $ip_head $redis_password
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $ip_head $redis_password &
sleep 5
##############################################################################################

#### call your code below
python examples/mnist_pytorch_trainable.py --cuda
exit
