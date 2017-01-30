#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "calcdb.h"
#include "treeminer.h"

using namespace std;

int *Dbase_Ctrl_Blk::FreqIdx=NULL;
int *Dbase_Ctrl_Blk::FreqMap=NULL;
int Dbase_Ctrl_Blk::MaxTransSz=0;
int Dbase_Ctrl_Blk::TransSz=0;
int *Dbase_Ctrl_Blk::TransAry=NULL;
int Dbase_Ctrl_Blk::Tid=0;
int Dbase_Ctrl_Blk::Cid=0;
int Dbase_Ctrl_Blk::NumF1=0;
int *Dbase_Ctrl_Blk::PvtTransAry=NULL;
bool Dbase_Ctrl_Blk::binary_input=false;
int** Dbase_Ctrl_Blk::DB_array=NULL;
int** Dbase_Ctrl_Blk::DB_vertical=NULL;
int** Dbase_Ctrl_Blk::DB_vertical_ref=NULL;
int Dbase_Ctrl_Blk::db_iter=0;
scope** Dbase_Ctrl_Blk::db_scope=NULL;
int Dbase_Ctrl_Blk::DB_array_size = 0;

//GPU
int* Dbase_Ctrl_Blk::trees_h = NULL;
int* Dbase_Ctrl_Blk::tr_start_ind_h = NULL;
int* Dbase_Ctrl_Blk::blk_ind_h = NULL;
int* Dbase_Ctrl_Blk::blk_sze_h = NULL;
int* Dbase_Ctrl_Blk::blk_tree_index_h = NULL;
int* Dbase_Ctrl_Blk::blk_tree_count_h = NULL;
int Dbase_Ctrl_Blk::blk_count_h=0;

vector<int> Dbase_Ctrl_Blk::DB_1D_array_sharedMem_starting(0);


Dbase_Ctrl_Blk::Dbase_Ctrl_Blk(const char *infile, const int buf_sz)
{
   if (binary_input){
      fd.open(infile, ios::in|ios::binary);
      if (!fd){
         cerr << "cannot open infile" << infile << endl;
         exit(1);
      }
   }
   else{
      fd.open(infile, ios::in);
      if (!fd){
         cerr << "cannot open infile" << infile << endl;
         exit(1);
      }
   }

   buf_size = buf_sz;
   buf = new int [buf_sz];
   cur_buf_pos = 0;
   cur_blk_size = 0;
   readall = 0;
   fd.seekg(0,ios::end);
   endpos = fd.tellg();
   fd.seekg(0,ios::beg);
}
   
Dbase_Ctrl_Blk::~Dbase_Ctrl_Blk()
{
   delete [] buf;
   fd.close();
}

void Dbase_Ctrl_Blk::get_first_blk()
{
   readall=0;

   fd.clear();
   fd.seekg(0,ios::beg);
   
   if (binary_input){
      fd.read((char *)buf, (buf_size*ITSZ));
      cur_blk_size = fd.gcount()/ITSZ; 
      if (cur_blk_size < 0){
         cerr << "problem in get_first_blk" << cur_blk_size << endl;
      }
      if (cur_blk_size < buf_size){
         fd.clear();
         fd.seekg(0,ios::end);
      }
   }
   
   cur_buf_pos = 0;
}

int Dbase_Ctrl_Blk::get_next_trans ()
{
   static char first=1;  

   if (first){
      first = 0;
      get_first_blk();
   }

   if (binary_input){
      if (cur_buf_pos+TRANSOFF >= cur_blk_size ||
          cur_buf_pos+buf[cur_buf_pos+TRANSOFF-1]+TRANSOFF > cur_blk_size){
         fd.seekg(0,ios::cur);
         if (((int) fd.tellg()) == endpos) readall = 1;      
         if (!readall){
            // Need to get more items from file
            get_next_trans_ext();
         }      
      }
      
      if (eof()){
         first = 1;
         return 0;
      }                     
      
      if (!readall){
         Cid = buf[cur_buf_pos];
         Tid = buf[cur_buf_pos+TRANSOFF-2];
         TransSz = buf[cur_buf_pos+TRANSOFF-1];
         TransAry = buf + cur_buf_pos + TRANSOFF;
         cur_buf_pos += TransSz + TRANSOFF;
      }
      return 1;
   }
   else{
      //fd.seekg(0,ios::cur);
      //if (fd.tellg() == endpos) readall = 1;

      if ((int)fd.tellg() == endpos-1){
         readall = 1;
         first = 1;
         return 0;
      }
      else{
         int i;
         fd >> Cid;
         fd >> Tid;
         fd >> TransSz;
         for (i=0; i < TransSz; ++i){
            fd >> buf[i];
         }
         TransAry = buf;
         cur_buf_pos = 0;

         //cout << "ENDPOS " << fd.tellg() << " " << endpos << endl;

         return 1;
      }
   }   
}

void Dbase_Ctrl_Blk::get_next_trans_ext()
{
   // Need to get more items from file
   int res = cur_blk_size - cur_buf_pos;
   if (res > 0)
   {
      // First copy partial transaction to beginning of buffer
      for (int i=0; i < res; i++)
         buf[i] = buf[cur_buf_pos+i]; 
      cur_blk_size = res;
   }
   else
   {
      // No partial transaction in buffer
      cur_blk_size = 0;
   }

   fd.read((char *)(buf + cur_blk_size),
           ((buf_size - cur_blk_size)*ITSZ));
   
   res = fd.gcount();
   if (res < 0){
      cerr << "in get_next_trans_ext" << res << endl;
   }
   
   if (res < (buf_size - cur_blk_size)){
      fd.clear();
      fd.seekg(0,ios::end);
   }

   cur_blk_size += res/ITSZ;
   cur_buf_pos = 0;
}


//Elaheh: add transaction size to the beginning of the TransAry for making DB_array
void Dbase_Ctrl_Blk::get_valid_trans()
{
   int i,j;
   const int invalid=-3; //-3 does not appear in original trans
   
   if (PvtTransAry == NULL)
      PvtTransAry = new int [MaxTransSz];
   
   //remove infreq items
   //cout << "ORIG " << endl;
   //for (i=0; i < TransSz; i++)
   //   cout << TransAry[i] << " ";
   //cout << endl;
   //cout << "ORIGV " << endl;
   //for (i=0; i < TransSz; i++)
   //   if (TransAry[i] != -1 && theFreqMap[TransAry[i]] == -1)
   //      cout << "-3 ";
   //   else   cout << TransAry[i] << " ";
   //cout << endl;
   

   for (i=0; i < TransSz; i++){      
      if (TransAry[i] != invalid){
         if (TransAry[i] != BranchIt){   
            if (FreqMap[TransAry[i]] == -1){
               //set item to invalid and the next -1 to invalid
               TransAry[i] = invalid;
               int cnt=0;
               
               for(j=i+1; j < TransSz && cnt >= 0; j++){
                  if (TransAry[j] != invalid){
                     if (TransAry[j] == BranchIt) cnt--;
                     else cnt++;
                  }
               }
               TransAry[--j] = invalid;
            }
         }
      }
   }
   
   //cout << "VALID " << endl;
   //for (i=0; i < TransSz; i++)
   //   cout << TransAry[i] << " ";
   //cout << endl;

   //copy valid items to PvtTransAry
   for (i=0,j=0; i < TransSz; i++){
      if (TransAry[i] != invalid){
         if (TransAry[i] == BranchIt) PvtTransAry[j] = TransAry[i];
         else PvtTransAry[j] = FreqMap[TransAry[i]];
         j++;
      }
   }
   TransAry = PvtTransAry;
   TransSz = j;

   //cout << "NEW " << endl;
   //for (i=0; i < TransSz; i++)
   //   cout << TransAry[i] << " ";
   //cout << endl;
}

void Dbase_Ctrl_Blk::print_trans(){
  cout << Cid << " " << Tid << " " << TransSz;
  for (int i=0; i < TransSz; i++)
    cout << " " << TransAry[i];
  cout << endl;
}

//Elaheh
struct vert_node{
	int label;
	int position;
};

void Dbase_Ctrl_Blk::create_vertical(int* tree, int*& vert_tree, int*& vert_tree_ref, scope*& db_scope) {
	int j = 1;
	deque<vert_node> q;
	int previous_isBranch = 0;
	vert_node* node;
	int scope_upperBound = -1;

	/*for(int t=0; t<tree[0]+1; t++) {
			cout << tree[t] << " ";
		}
	cout << endl;*/

	for (int i = 1; i < tree[0] + 1; i++) {
		if (tree[i] != BranchIt) {
			node = new vert_node;
			node->label = tree[i];
			node->position = i;
			q.push_back(*node);
			scope_upperBound++;
			db_scope[i].begin = scope_upperBound;
			previous_isBranch = 0;
			if (i == tree[0]) {
				for (deque<vert_node>::iterator k = q.begin(); k != q.end(); k++) {
					vert_tree[j] = (*k).label;
					vert_tree_ref[j] = (*k).position;
					j++;
				}
			}
		} else if (tree[i] == BranchIt && !previous_isBranch) {
			for (deque<vert_node>::iterator k = q.begin(); k != q.end(); k++) {
				vert_tree[j] = (*k).label;
				vert_tree_ref[j] = (*k).position;
				j++;
			}
			previous_isBranch = 1;
			db_scope[q.back().position].end = scope_upperBound;
			q.pop_back();
			vert_tree[j] = BranchIt;
			vert_tree_ref[j]=-1;
			j++;
		} else if (tree[i] == BranchIt && previous_isBranch) {
			db_scope[q.back().position].end = scope_upperBound;
			q.pop_back();
		}
	}

	db_scope[1].end = scope_upperBound;
	vert_tree[0] = j-1;
	vert_tree_ref[0] = j-1;

	/*for(int t=0; t<vert_tree[0]+1; t++) {
		cout << vert_tree[t] << " ";
	}
	cout << endl;

	for(int t=0; t<vert_tree_ref[0]+1; t++) {
			cout << vert_tree_ref[t] << " ";
		}
		cout << endl;
	for(int t=0; t<tree[0]+1; t++) {
				cout << "(" << db_scope[t].begin << "," << db_scope[t].end << ")" << endl;
			}
			cout << endl;*/
}

