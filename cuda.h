#ifndef CUDA_H
#define CUDA_H

#include <stdint.h>

#define CONSTANT_MEMORY_SIZE 16384
typedef int NodeType;
typedef int16_t StackType; //TODO calculate it based on the dataset

//__constant__ NodeType C[16384];


typedef struct {
	NodeType* labels;
	int size;

} Tree;

typedef struct {
	StackType index_in_tree;
	StackType diff_height;
	StackType match_index;
}MatchData;


#include <assert.h>
#include<driver_types.h>

#define BACK_TRACK -1





#endif
