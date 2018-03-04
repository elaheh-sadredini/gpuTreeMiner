# GPU-TreeMiner
BFS implementation of frequent subtree mining on the GPU

 -- having the candidates in the shared memory 
 -- implemented based-on Zaki's implementation

## Usage:

`1. make`

`2. ./gsp_cpu <frequency> <input file> <Dumping results (yes = 1, No = 0)> <Allow gap between itemsets (yes = 1, No = 0)>`
--weighted, prune, ...

You can set "Allow gap between itemsets" to "0" in order to mine the frequent consequtive itemsets. 



## Input format:

**Sample input file**

        1 2 -1 3 4 -1 -2
        5 6 -1 -2
        -2
        
'-1' is a delimiter between itemsets.

'-2' is a delimiter betweeen sequences.

'-2' should be added to the last line.
        
        
        
**Sample output file**

        1-2--1-3
        5--1


'-' is a delimiter to separate items in one itemset (subsequence).
'--' is a delimiter to separate itemsets in the sequence.



## Contact:

elaheh@virginia.edu


## Citations:
Please cite the following papers if you are using this tool for your research. 

[\[1\] Elaheh Sadredini, Reza Rahimi, Ke Wang, and Kevin Skadron. "Frequent Subtree Mining on the Automata Processor: Opportunities
and Challenges." ACM International Conference on Supercomputing (ICS), Chicago, June 2017](http://www.cs.virginia.edu/~skadron/Papers/sadredini_ics17.pdf) 

