#ifndef __DATABASE_H
#define __DATABASE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <deque>
#include <map>
#include <ext/hash_map>
#include <list>

using namespace std;

#define ITSZ sizeof(int)
#define DCBBUFSZ 2048
#define TRANSOFF 3

struct scope{
	int begin;
	int end;
};

class Dbase_Ctrl_Blk{
private:
   ifstream fd;
   int buf_size;
   int * buf;
   int cur_blk_size; 
   int cur_buf_pos;
   int endpos;
   char readall;
   
   static int *PvtTransAry;

public:
   static int NumF1;   //number of freq items
   static int *FreqMap; //mapping of freq items, i.e., item to freq_idx
   static int *FreqIdx; //freq_idx to original item value

   static int *TransAry;
   static int TransSz;
   static int Tid;
   static int Cid;
   static int MaxTransSz;
   static bool binary_input;
   static int** DB_array; //the first element is the size of the array
   static int DB_array_size; //the size of DB_array, it is needed for creating GPU dataset 
   static int** DB_vertical; //the first element is the size of vertical tree
   static int** DB_vertical_ref; //reference to the original location in the maim dataset
   static int db_iter;
   static scope** db_scope; //the first element does not show anything
   
    //GPU
    static int* trees_h; // The whole dataset in one dimensional array for the GPU
    static int* tr_start_ind_h; //start of the trees in "trees_h". It also consider the size of the tree for each transaction
    static int* blk_ind_h; //This array contains the start index of bulks. Length of this array is equal to the number of bulks. The value of each cell is an index to trees.
    static int* blk_sze_h; //This array contains the bulk sizes
    static int* blk_tree_index_h; //This array maps the start tree index of bulk
    static int* blk_tree_count_h; //This array contains number of trees in each bulk
    static int blk_count_h; //This parameter is the total number of bulks
    
    static vector<int> DB_1D_array_sharedMem_starting; //Starting position of each chunk of data (to be copied on the shared memory) in the one dimensional database array
   

   Dbase_Ctrl_Blk(const char *infile, const int buf_sz=DCBBUFSZ);
   ~Dbase_Ctrl_Blk();
   
   void get_next_trans_ext();
   void get_first_blk();
   int get_next_trans();
   void get_valid_trans();
   void print_trans();
   int eof(){return (readall == 1);}
   void create_vertical(int* tree, int*& vert_tree,int*& vert_tree_ref, scope*& db_scope);
   void get_vertical_DB();
   
};

#endif //__DATABASE_H





