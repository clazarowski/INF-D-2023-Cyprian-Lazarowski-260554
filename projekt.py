import time
import numpy
import os, glob
import pygad

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

def count_score(id_tab, solution, num):
    count = 0
    for i in id_tab:
        if solution[i]==1:
            count+=1
    return abs(num-count)

def fintess_func_factory(x,y,fill_a_pix):
    def fitness_func(solution, solution_idx):
        mistake = 0
        for i in range(y):
            for j in range(x):
                if fill_a_pix[i][j]!="-":
                    num = int(fill_a_pix[i][j])
                    id_tab = get_all_neighborhood_id(i,j,x,y)
                    mistake+=count_score(id_tab,solution,num)
        
        return (x*y*9)-mistake
    return fitness_func

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

def genetic_algorithm(x, y, fill_a_pix):
    ga_instance = pygad.GA(gene_space=[0,1],
        num_generations=500,
        num_parents_mating=5,
        fitness_func=fintess_func_factory(x,y,fill_a_pix),
        sol_per_pop=20,
        num_genes=x*y,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type = "random",
        mutation_percent_genes = 10,
        stop_criteria=[f"reach_{x*y*9}"],
        )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print_solution(solution, y, x)
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print(f"Max score: {x*y*9}")
    ga_instance.plot_fitness()

filenames = filter_filename(get_example_filename())
for filename in filenames:
    fill_a_pix = get_fill_a_pix('./fill-a-pix/'+filename)
    x,y=get_len(fill_a_pix)
    print(filename, x, y)
    genetic_algorithm(x,y,fill_a_pix)