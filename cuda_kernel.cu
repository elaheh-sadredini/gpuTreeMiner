/* Optimizations:
    1. Sorting trees
    2.
*/


#include "cuda.h"
#include <cuda_runtime.h>
#include "stdio.h"

using namespace std;

__constant__ NodeType C[16384];

static __device__ int find_count(Tree& tr, Tree& can, MatchData* data) {

	int cand_size = (can.size + 1) / 2;

	StackType tr_index = 0;
	StackType can_index = 0;

	StackType matched_count = 0;
	StackType current_height = 0;

	data[0].diff_height = -1;
	data[0].index_in_tree = -1;
	data[0].match_index = -1;

	int* tree_labels = tr.labels;
	int* cand_labels = can.labels;

	int count = 0;

	while (!(matched_count==0 && tr_index==tr.size)) {
		if (tr_index == tr.size) {
			matched_count--;
			tr_index = data[matched_count].index_in_tree + 1;
			can_index = data[matched_count].match_index;
			current_height = data[matched_count].diff_height;

		} else if (cand_labels[can_index] == BACK_TRACK) {

			StackType height_to_return = 0;
			int i = 0;
			while (cand_labels[can_index + i] == BACK_TRACK) {
				height_to_return += data[matched_count - i].diff_height;
				i++;
			}
			current_height = 0;

			while ((-current_height <= height_to_return) && (tr_index < tr.size)) {
				if (tree_labels[tr_index] == BACK_TRACK)
					current_height--;
				else
					current_height++;

				tr_index++;
			}
			if (tr_index == tr.size) {
				continue;
			}

			current_height = data[matched_count - i].diff_height-1;
			can_index += i;
		}
		else if (tree_labels[tr_index] == BACK_TRACK) {
			tr_index++;
			current_height--;
			if (current_height == -1) {

				matched_count--;
				tr_index = data[matched_count].index_in_tree + 1;
				can_index = data[matched_count].match_index;
				current_height = data[matched_count].diff_height;


				continue;

			}
		}
		else if (tree_labels[tr_index] == cand_labels[can_index]) {

			current_height++;
			data[matched_count].diff_height = current_height;
			data[matched_count].index_in_tree = tr_index;
			data[matched_count].match_index = can_index;


			if (matched_count + 1 == cand_size) {
				//matched_count++;
				count++;
				return 1;
			} else {
				matched_count++;
				can_index++;
				current_height = 0;

				data[matched_count].diff_height = 0;
				data[matched_count].index_in_tree = 0;
				data[matched_count].match_index = 0;

			}

			tr_index++;

		} else if (tree_labels[tr_index] != cand_labels[can_index]) {
			current_height++;
			tr_index++;
		}

	}

	return count;

}

//static __device__ int find_count(Tree& tr, Tree& can, MatchData* data) {
//
//	int cand_size = (can.size + 1) / 2;
//
//	int tr_index = 0;
//	int can_index = 0;
//
//	int matched_count = 0;
//	int current_height = 0;
//
//	data[0].diff_height = -1;
//	data[0].index_in_tree = -1;
//	data[0].match_index = -1;
//
//	int* tree_labels = tr.labels;
//	int* cand_labels = can.labels;
//
//	int count = 0;
//
//	while (!(matched_count==0 &&tr_index==tr.size)) {
//		if (tr_index == tr.size) {
//			matched_count--;
//			tr_index = data[matched_count].index_in_tree + 1;
//			can_index = data[matched_count].match_index;
//			current_height = data[matched_count].diff_height;
//
//		} else if (cand_labels[can_index] == BACK_TRACK) {
//
//			int height_to_return = 0;
//			int i = 0;
//			while (cand_labels[can_index + i] == BACK_TRACK) {
//				height_to_return += data[matched_count - i].diff_height;
//				i++;
//			}
//			current_height = 0;
//
//			while ((-current_height <= height_to_return) && (tr_index < tr.size)) {
//				if (tree_labels[tr_index] == BACK_TRACK)
//					current_height--;
//				else
//					current_height++;
//
//				tr_index++;
//			}
//			if (tr_index == tr.size) {
//				continue;
//			}
//
//			current_height = data[matched_count - i].diff_height-1;
//			can_index += i;
//		}
//		else if (tree_labels[tr_index] == BACK_TRACK) {
//			tr_index++;
//			current_height--;
//			if (current_height == -1) {
//
//				matched_count--;
//				tr_index = data[matched_count].index_in_tree + 1;
//				can_index = data[matched_count].match_index;
//				current_height = data[matched_count].diff_height;
//
//
//				continue;
//
//			}
//		}
//		else if (tree_labels[tr_index] == cand_labels[can_index]) {
//
//			current_height++;
//			data[matched_count].diff_height = current_height;
//			data[matched_count].index_in_tree = tr_index;
//			data[matched_count].match_index = can_index;
//
//
//			if (matched_count + 1 == cand_size) {
//				//matched_count++;
//				count++;
//			} else {
//				matched_count++;
//				can_index++;
//				current_height = 0;
//
//				data[matched_count].diff_height = 0;
//				data[matched_count].index_in_tree = 0;
//				data[matched_count].match_index = 0;
//
//			}
//
//			tr_index++;
//
//		} else if (tree_labels[tr_index] != cand_labels[can_index]) {
//			current_height++;
//			tr_index++;
//		}
//
//	}
//
//	return count;
//
//}

/**
 *
 * @param trees This parameter is an array representating all the trees in the DB. it starts with the size of
 * a tree and then representing the lables and then next tree. for example,{5,A,B,-1,C-1} representing the following tree:   A
 * 																															/ \
 * 																														   B   C
 * @param ta_start_ind This array is a way to find the start point of trees. The index contains the size too. for example if we have
 * two replica of the previous tree, this array would be something like this {0,6}
 *
 * @param tr_cnt This parameter contains the total number of trees.
 *
 * @param cand_size This parameter specifies the number of nodes in candidate.
 *
 * @cand_cnt This parameter contains total number of generated candidates.
 *
 * @blk_ind This array contains the start index of bulks. Length of this array is equal to the number of bulks. The value of each cell is
 * an index to trees.
 *
 * @blk_sze This array contains the bulk sizes
 *
 * @blk_count This parameter is the total number of bulks
 *
 * @blk_tree_count This array contains number of trees in each bulk
 *
 * @blk_tree_index This array maps the start tree index of bulk
 *
 * @freq_result This parameter is the output. Its size should be equal to the candidates count
 */

static __global__ void frequency_counter(NodeType *trees, int* tr_start_ind,
		int tr_cnt, int cand_size, int cand_cnt, int* cand, int* freq_result) {

	__shared__ StackType curr_blk[24176];
	__shared__ int cand_sharedMem[200];

		for (int c = 0; c < cand_cnt; c++) {

			//copy the candidate to the shared memory
			if(threadIdx.x < 2*cand_size-1){
				cand_sharedMem[threadIdx.x] = cand[((2*cand_size-1)*c) + threadIdx.x];
			}

			int curr_tree_index = threadIdx.x + blockIdx.x * blockDim.x;

			while (curr_tree_index < tr_cnt) {

//				NodeType * under_process_tree =
//						&curr_blk[tr_start_ind[curr_tree_index_in_bulk
//								+ curr_bulk_start_tree_index] - curr_blk_str];

				NodeType * under_process_tree =
						&trees[tr_start_ind[curr_tree_index]];

				Tree t;
				t.size = under_process_tree[0];
				t.labels = &under_process_tree[1];

				Tree can;
				can.size=2*cand_size-1;
				can.labels=cand_sharedMem;

//				int res=find_count(t,can,(MatchData*) &curr_blk[curr_blk_sze+threadIdx.x*3*cand_size]);
				int res=find_count(t,can,(MatchData*) &curr_blk[threadIdx.x*3*cand_size]);

				if(res>0) res = 1;
				atomicAdd(&freq_result[c],res);

				//curr_tree_index_in_bulk += blockDim.x;
				curr_tree_index += gridDim.x * blockDim.x;
				__syncthreads();
			}

		}

//		curr_blk_ind += gridDim.x;
//		__syncthreads();

}

