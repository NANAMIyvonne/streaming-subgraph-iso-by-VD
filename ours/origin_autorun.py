import os
import subprocess
import concurrent.futures

datasets = ['yeast', 'human', 'cora', 'citeseer', 'pubmed', 'wordnet']
nodes = [10, 15, 20, 25, 30]

def run_command(dataset, node):
    prefix = f"../data/{dataset}_nnode{node}_k50_coldstart"
    
    coldstart_file = f"{dataset}_nnode{node}_coldstart.txt"
    output_file = f"{dataset}_nnode{node}.txt"
    
    # command for coldstart
    command1 = f"./origin -t 0.9 -e 100 -c -s 2000 {prefix}/d.dimas {prefix}/query/ > {coldstart_file}"
    
    # command for other task
    command2 = f"./origin -t 0.9 -e 100 -c -s 2000 {prefix}/d.dimas {prefix}/query/ {prefix}/query/ > {output_file}"
    
    subprocess.run(command1, shell=True, check=True)
    subprocess.run(command2, shell=True, check=True)

# Use a ThreadPoolExecutor to run the commands in parallel.
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    for dataset in datasets:
        for node in nodes:
            executor.submit(run_command, dataset, node)
