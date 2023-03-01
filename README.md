# CBSH2-RTC-CHBP
An improvement technique of Conflict-Based Search (CBS) [1] for Multi-Agent Path Finding (MAPF).
CHBP reasons the conflicts beyond the two agents and allow us to (i) generate stronger heuristics; and (ii) explore more bypasses. 
CHBP can seamlessly integrate with the current state-of-the-art solver CBSH2-RTC. 
Experimental results show that CHBP improves CBSH2-RTC by solving more instances. 
On the co-solvable instances, CHBP runs faster than CBSH2-RTC with speedups ranging from several factors to over one order of magnitude.

 
## Requirements 
The implementation requires the external libraries: [CMake](https://cmake.org) and [Boost](https://www.boost.org/). 


If you are using Ubuntu, you can install them simply by:
```shell script
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
``` 
If you are using Mac, you can install them simply by:
```shell script
brew install cmake
brew install boost
```
If the above methods do not work, you can also follow the instructions
on the [CMake](https://cmake.org) or [Boost](https://www.boost.org/) website and install it manually.



## Compiling and Running
The current implementation is compiled with CMake, you can compile it from the directory of the source code by:
```shell script
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make -j
```

Our implementation is based on [CBSH2-RTC](https://github.com/Jiaoyang-Li/CBSH2-RTC), the leading optimal solver for
MAPF. By default, CBSH2-RTC runs the best variant of the code reported in [2] (i.e., using
prioritizing conflicts,
bypassing conflicts,
WDG heuristics,
target reasoning, and
generalized rectangle and corridor reasoning).
On top of this best variant, our implementation runs from a new flag "--cluster_heuristics", which contains four options:

* N: CBSH2-RTC (without any modifications).
* CH: CBSH2-RTC + Cluster Heuristic only.
* BP: CBSH2-RTC + Bypass only.
* CHBP: CBSH2-RTC + Cluster Heuristic and Bypass (final algorithm).

Our final algorithm CHBP runs by:
```shell script
./cbs -m random-32-32-20.map -a random-32-32-20-random-1.scen -o test.csv -k 30 -t 60 --cluster_heuristic=CHBP
```
- m: the map file from the MAPF benchmark
- a: the scenario file from the MAPF benchmark
- o: the output file that contains the search statistics
- k: the number of agents
- t: runtime limit (in seconds)
- --cluster_heuristic: our new improvement algorithm CHBP.



## Dataset
To test the code on more instances or easily reproduce our experiments, 
we include the MAPF instances downloaded from the [MAPF benchmark](https://movingai.com/benchmarks/mapf/index.html).
In particular, the format of the scen files is explained [here](https://movingai.com/benchmarks/formats.html).
For a given number of agents k, the first k rows of the scen file are used to generate the k pairs of start and target locations.

All maps and scenario files are included in the "/dataset" folder. 



## Guideline
Here, we give a short guideline in order to access our codes:

- Our implementation mainly modifies the following files:
  - "/inc/CBSHeuristic.h"
  - "/src/CBSHeuristic.cpp" 
- According to our paper, the pseudo-code of algorithms indicate the following functions in our implementation:
  - Algorithm 1: computeClusterHeuristicAndBypass()
  - Algorithm 2: findClusterOrBypass()
- We provide bash scripts that automatically run all experiments reported in our paper.
The bash script also creates "/results" folder under the current directory, all results will appear in this folder.
To reproduce our experiment, please run: 
  - ```shell script
    bash ./run_all_experiments.sh
    ```
- To visualize the experimental results or reproduce the plots in our paper, 
we provide bash and python scripts. For python scripts, we require the external libraries: [pandas](https://pandas.pydata.org), 
[NumPy](https://numpy.org), [matplotlib](https://matplotlib.org) and [jupyter](https://jupyter.org).
Please install them properly. Once installed, please go to "/analysis" folder:
  - run bash scripts to merge all experimental results.
      ```shell script
      bash ./merge_results.sh
      ```
  - use python scripts in "/analysis/experiments.ipynb" to generate plots to "/analysis/fig"
      ```shell script
      jupyter notebook experiments.ipynb
      ```
Contact
===========================================================
For any question, please contact Bojie.Shen@monash.edu.

[//]: # (**Note that the license is removed for anonymous purposes. Please do not distribute the codes.**)
## References

[1] Guni Sharon, Roni Stern, Ariel Felner, and Nathan R. Sturtevant.
Conflict-Based Search for Optimal Multi-Agent Pathfinding.
Artificial Intelligence, 219:40–66, 2015.

[2] Jiaoyang Li, Daniel Harabor, Peter J. Stuckey, and Sven Koenig.
Pairwise Symmetry Reasoning for Multi-Agent Path Finding Search.
Artificial Intelligence, 301: 103574, 2021.
