import time
import numpy as np
import os, glob
import pygad
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_cost_history
import matplotlib.pyplot as plt
import statistics

#Read files
def get_example_filename(path='./fill-a-pix'):
    for root, dirs, files in os.walk(path, topdown=False):
        return files

def filter_filename(files):
    examples = []
    for filename in files:
            if filename.endswith('.txt'):
                examples+=[filename]
    return examples

#Get fill_a_pix
def get_fill_a_pix(filename):
    fill_a_pix = []
    with open(filename) as f:
        lines = f.read().splitlines()
        for line in lines:
            fill_a_pix.append(list(line))
    return fill_a_pix

def get_len(fill_a_pix):
    return len(fill_a_pix[0]), len(fill_a_pix)

def count_all_with_num(fill_a_pix):
    with_num = 0
    for row in fill_a_pix:
        for num in row:
            if num!="-":
                with_num+=1

    return with_num

#fitness_func
def count_score(id_tab, solution, num, option):
    count = 0
    for i in id_tab:
        if solution[i]==1:
            count+=1
    if option==0: return abs(num-count)
    elif option==1:
        if num == count: return 0
        else: return 1

def fintess_func_factory(x,y,fill_a_pix,calculate_max_result,option):
    def fitness_func(solution, solution_idx):
        mistake = 0
        for i in range(y):
            for j in range(x):
                if fill_a_pix[i][j]!="-":
                    id_tab = get_all_neighborhood_id(i,j,x,y)
                    mistake+=count_score(id_tab,solution,int(fill_a_pix[i][j]),option)
        
        return calculate_max_result-mistake
    return fitness_func

def fitness_func_PSO(solution,x,y,fill_a_pix,option):
    mistake = 0
    for i in range(y):
        for j in range(x):
            if fill_a_pix[i][j]!="-":
                id_tab = get_all_neighborhood_id(i,j,x,y)
                mistake+=count_score(id_tab,solution,int(fill_a_pix[i][j]),option)
        
    return mistake

def get_all_neighborhood_id(i,j,x,y):
    id_tab = []
    id = (i*x)+j

    if i>0:
        id_up = ((i-1)*x)+j
        if j>0: id_tab+=[id_up-1]
        id_tab+=[id_up]
        if j<x-1: id_tab+=[id_up+1]
                
    if j>0: id_tab+=[id-1]
    id_tab+=[id]
    if j<x-1: id_tab+=[id+1]

    if i<y-1:
        id_down = ((i+1)*x)+j
        if j>0: id_tab+=[id_down-1]
        id_tab+=[id_down]
        if j<x-1: id_tab+=[id_down+1]

    return id_tab

def print_solution(solution, y, x):
    print("The Best Solution:")
    for i in range(y):
        for j in range(x):
            print(int(solution[(i*x)+j]), end=' ')
        print("")

def genetic_algorithm(x, y, fill_a_pix,max_result,option,print_data):
    start = time.time()
    ga_instance = pygad.GA(gene_space=[0,1],
        num_generations=10000,
        num_parents_mating=5,
        fitness_func=fintess_func_factory(x,y,fill_a_pix,max_result,option),
        sol_per_pop=20,
        num_genes=x*y,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type = "random",
        mutation_percent_genes = 1,
        stop_criteria=[f"reach_{max_result}"],
        )

    ga_instance.run()
    end_time = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if print_data == 1:
        print_solution(solution, y, x)
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print(f"Max score: {max_result}")
        print(f"Time: {end_time-start}")
        ga_instance.plot_fitness()
        return
    else: return end_time-start, solution_fitness

def fitness_PSO(x, **kwargs):
    n_particles = x.shape[0]
    j = [fitness_func_PSO(x[i], kwargs.get('z'), kwargs.get('y'), kwargs.get('fill_a_pix'), kwargs.get('option')) for i in range(n_particles)]
    return np.array(j)

def PSO_algorithm(x,y,fill_a_pix, max_resoult, option):
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}
    kwargs = {'z': x, 'y': y, 'fill_a_pix': fill_a_pix, 'max': max_resoult, 'option': option}
    optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=x*y, options=options)
    start = time.time()
    optimizer.optimize(fitness_PSO, iters=10000, verbose=True, **kwargs)
    end = time.time()
    print(f"Time for algorithm: {end-start}")
    plot_cost_history(optimizer.cost_history)
    plt.show()

def test_100_times(x,y,fill_a_pix, max_resoult, option, with_print):
    program_times=[]; solutions=[]; count_wrong=0
    for i in range(100):
        program_time, solution = genetic_algorithm(x,y,fill_a_pix,max_resoult,option,with_print)
        print(f"Program time: {program_time}, Solution: {solution}")
        if solution!=max_resoult: count_wrong+=1
        program_times.append(program_time); solutions.append(solution)

    print(f"All: 100, Find: {100-count_wrong}, Not find: {count_wrong}, Mean: {statistics.mean(solutions)}")
    print(f"Time Mean: {statistics.mean(program_times)}, Max: {max(program_times)}, Min: {min(program_times)}")

filenames = filter_filename(get_example_filename())
ex_to_test = 0
for filename in filenames:
    fill_a_pix = get_fill_a_pix('./fill-a-pix/'+filename)
    x,y=get_len(fill_a_pix)
    
    print("================================")
    print(f"Example: {filename} ({x}x{y})")
    print("================================")
    print(f"Genetic Algorithm (option 1): ")
    genetic_algorithm(x,y,fill_a_pix,x*y,0,1)
    print("================================")
    print(f"Genetic Algorithm (option 2): ")
    genetic_algorithm(x,y,fill_a_pix,count_all_with_num(fill_a_pix),1,1)
    print("================================")
    print(f"PSO Algorithm (option 1):")
    PSO_algorithm(x,y,fill_a_pix,x*y,0)
    print("================================")
    print(f"PSO Algorithm (option 2):")
    PSO_algorithm(x,y,fill_a_pix,count_all_with_num(fill_a_pix),1)
    print("================================")
    
    if ex_to_test!=0:
        print(f"Genetic Algorithm (option 1) x 100: ")
        test_100_times(x,y,fill_a_pix,x*y,0,0)
        print("================================")
        print(f"Genetic Algorithm (option 2) x 100: ")
        test_100_times(x,y,fill_a_pix,count_all_with_num(fill_a_pix),1,0)
        print("================================")
        ex_to_test-=1
    