# GPUTreeMiner
BFS implementation of frequent subtree mining on the GPU. It uses shared memory for the generated subtree candidates. GPUTreeMiner provides pruning huristics and both weighted (counts a candidate subtree multiple times in a tree) and unique_count options (counts a candidate subtree once in a tree). 


This code uses [TreeMiner](http://www.cs.rpi.edu/~zaki/www-new/pmwiki.php/Software/Software#toc14) as the base for CPU implementation. 

## Usage:

`1. make`

`2. ./gpuTreeMiner -i<input_file> -s<support> -o<print output> -p<prune> -u<unique counting>`

        -i,      dataset of trees
        -s,      support threshold between (0,1)
        -o,      <True> if printing the freuqnt subtrees. Default is <False>
        -p,      <True> if pruning the database, <False> otherwise. Default is <True>
        -u,      <True> if counting the subtree matches once per tree, <False> if weighted counting. Default is <True>


You can set "Allow gap between itemsets" to "0" in order to mine the frequent consequtive itemsets. 



## Input format:
The input must be in the following format:

        id id length string_encoding

where id is repeated twice (the same value for the tree number), 
length is the number of items to follow on the line, and
string_encoding is the coding of the tree 


**Sample input file**

        0 0 7 1 2 -1 3 4 -1 -1
        1 1 11 2 1 2 -1 4 -1 -1 2 -1 3 -1
        2 2 15 1 3 2 -1 -1 5 1 2 -1 3 4 -1 -1 -1 -1
        
Trees are in pre-order travesal and -1 shows a backtrack. 

The first tree's string encoding has length 7 (including -1's), and so on.
This database has 3 trees.
        
 
## Contact:

elaheh@virginia.edu


## Citations:
Please cite the following papers if you are using this tool for your research. 

[\[1\] Elaheh Sadredini, Reza Rahimi, Ke Wang, and Kevin Skadron. "Frequent Subtree Mining on the Automata Processor: Opportunities
and Challenges." ACM International Conference on Supercomputing (ICS), Chicago, June 2017](http://www.cs.virginia.edu/~skadron/Papers/sadredini_ics17.pdf) 
