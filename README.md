# Multipartite_MMSBM

Program to find the membership factors and the probabilities of connections in a bipartite network adding metadata to the nodes. This program finds the most plausible parameters of the Mixed-Membership Stochastick Block Model, that fits a dataset of links between nodes. Also the user can add nodes' metadata that can help (or not) to the link prediction.
# Language
- Python >= 3.5

# Requirements
- pandas
- numpy
- numba
- PyYaml

# Configuration
When you want to perform an analysis, you have to create in the folder where the results will be stored a file named `config.yaml`. It has to be written in a yaml format with this structure:

```
folder: !!str Folder_directory

#Put {F} instead of the fold number
links:
    test: !!str formated_u{F}.test
    base: !!str formated_u{F}.base
    separator_base: !!str §
    separator_test: !!str \t
    rating_header: !!str rating

nodes:
    nodes_header: !!str userId
    nodes_meta:
    - metadata_id_1
    - metadata_id_2
    ...
    file: !!str formated_users_tmp2.dat
    K: !!int 10
    separator: !!str \t
    lambda_nodes: 0.01

items:
    items_meta:
    - metadata_id_1
    - metadata_id_2
    ...
    items_header: !!str movieId
    file: !!str formated_movies_tmp2.dat
    L: !!int 10
    separator: !!str \t
    lambda_items: 0.0
    Taus:
    - 10

seed: !!int 77777
N_fold: !!int 2

simulation:
    N_itt: !!int 3000
    N_measure: !!int 1
```
## `folder`
Directory where your dataset that you want to analyze is stored.

## `links`
Section where you have to put information about the file that contains the links:
- `test`: Filename of the test set file
- `base`: Filename of the base set file
- `separator_test` and `separator_base`: Separators of the test and base files (default `\t`)
- `rating_header`: Header of the rating column
If your filenames have a number referring to the fold, you can write `{F}` and the program will substitute it automatically for the fold number.
## `nodes`
Section where you put the information of your nodes metadata file
- `nodes_header`: Header of the nodes identification column
- `K`: Numbers of nodes groups
- `file`:Filename with nodes metadata's information
- `nodes_meta`: Vector with the name of all the  nodes' metadata headers'
- `separator`: Separators of the nodes' file (default `\t`)
- `lambda_items`: Intensity of nodes's priors

## `items`
Section where you put the information of your items metadata file
- `items_header`: Header of the items identification column
- `L`: Numbers of items groups
- `file`:Filename with items metadata's information
- `items_meta`: Vector with the name of all the  items' metadata headers'
- `separator`: Separators of the items' file (default `\t`)
- `lambda_items`: Intensity of items's priors
- `Taus`: Vector with the numbers of groups of items'

## `seed`
Integer with the seed that the algorithm will initialize the membership and probabilities matrices

## `simulation`
Information about the simulation procedure
- `N_itt` Number of iterationts
- `N_measure` Number of iterations of separation between convergence checking

# Files structure
To use the program you need, at less, the file that contains the links of the bipartite network that you want to infer. If you want to add metadata to the nodes or the items you need one file for each containing the information of each node/item. All files must be writed in columns. By default, the program reads the files using a tabulator (`\t`) as a separator. If you use another separator you can change it in the `config.yaml` file as detailed above. The information about nodes, items and metadata must be integers that starts from 0 and increase sequentially. Here an example of a file that contains links:

```
node_id item_id label_id
0   2   0
0   6   2
0   8   1
1   3   0
1   2   1
...

```
Note that the first row contains the headers that you can indicate in the `config.yaml`.

And here we have an example of metadata file, in this case, we add to the _item_id_ the _genre_id_ metadata.
```
item_id genre_id
0       2
1       2
2       3
3       0
4       4
5       0
6       0
7       0
8       1
9       4
10      4
11      1

```
# Use
To run the program you have to go to the directory where the `config.yaml` is and run the code.

`python3 Algoritme_model_2.py [OPTIONS]`

## Description
- `--help`,`-h` display this help and exit
-  `-K` number of user's groups that the MMBSM algorithm will user
- `-L`number of item's groups that the MMBSM algorithm will user
- `--lambda_nodes` value of the user's metadata visibility
- `--lambda_items` value of the item's metadata visibility
- `--seed`, `-s` seed to generate the random matrices. It is optional and must be an integer.
- `-N`, `--N_itt` number maximum of iterations that the maximization-expectation algorithm will perform.
- ``-n``, ``--N_meas`` number of steps between each record of data of the likelihood evolution.
- `--Redo`, `-R` If it is True, it redo simulation if it was done before, else, if the simulation was done before the program automatically exits.
- `--N_simu` optional label to add to the directory where the output files will be stored. If ``None`` the seed will appear instead.
- `--dir_format` directory format, adding information about the lambdes (``lambdes``) or the groups number (``groups``).
- `-F`,``--Fold`` fold number that appears in the links' filename in case that a _k_-fold cross validation is being used.

# Output

 Once the MAP algorithm starts, a directory to store the outputs files is created. The folder will be created in the same directory where the program was executed and has the following structure.

- `simu_ln_{lambda_nodes}_li_{lambda_items}/results_simu_s_{seed}_f_{Fold}`  
  if `--dir_format`=`lambdes`
- `simu_K_{K}_L_{L}/results_simu_s_{seed}_f_{Fold}`
  if `--dir_format`=``groups``

In this directory four different files will appear:
- d
