# MMSBM_wPriors

Program that uses the Mix-Membership Stochastick Block Model using metadata as prior.

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
Directory where your data that you want to analyze is stored.

## `links`
Section where you have to put information of the input filenames:
- `test`: Filename of the test set file
- `base`: Filename of the base set file
- `separator_test` and `separator_base`: Separators of the test and base files (default `\t`)
- `rating_header`: Header of the rating column
If your filenames have a number referring to the fold, you can write `{F}` and the program will change it automatically for the fold number.
## `nodes`
Section where you put the information of your nodes
- `nodes_header`: Header of the nodes identification column
- `K`: Numbers of nodes groups
- `file`:Filename with nodes metadata's information
- `nodes_meta`: Vector with the name of all the  nodes' metadata headers'
- `separator`: Separators of the nodes' file (default `\t`)
- `lambda_items`: Intensity of nodes's priors

## `items`
Section where you put the information of your items
- `items_header`: Header of the items identification column
- `L`: Numbers of items groups
- `file`:Filename with items metadata's information
- `items_meta`: Vector with the name of all the  items' metadata headers'
- `separator`: Separators of the items' file (default `\t`)
- `lambda_items`: Intensity of items's priors
- `Taus`: Vector with the numbers of groups of items'

## `seed`
Integer with the seed that the algorithm will initialize the membership and probabilities matrix

## `simulation`
Information about the simulation procedure
- `N_itt` Number of iterationts
- `N_measure` Number of iterations of separation between convergence checking

# Use
To run the program you have to go to the directory where the `config.yaml` is and run the code.
