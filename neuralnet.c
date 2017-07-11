// First attempt at making a feed-forward back-propogating neural network in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nn_common.h"

double inputs[4][2] = {0, 0,  0, 1,  1, 0,  1, 1};
double and_targets[4][1] = {0, 0, 0, 1};
double xor_targets[4][1] = {0, 1, 1, 0};
double or_targets[4][1] = {0, 1, 1, 1};
double nand_targets[4][1] = {1, 1, 1, 0};
double nor_targets[4][1] = {1, 0, 0, 0};
double** actual_target;

int main(int argc, char **argv)
{
    int input_count = 0;
    int pattern_count = 0;
    int hidden_count = 0;
    int output_count = 0;
    double eta = 0.0;
    int epochs = 0;

    actual_target = alloc_2d(4, 1);
    if (argc == 7)
    {
	input_count = atoi(argv[1]);
	pattern_count = atoi(argv[2]);
	hidden_count = atoi(argv[3]);
	output_count = atoi(argv[4]);
	eta = atof(argv[5]);
	epochs = atoi(argv[6]);
	memcpy(*xor_targets, *actual_target, 4 * sizeof(double));
    }
    else if (argc == 5)
    {
	// I don't care about optimizing the searches right now,
	// for now, this'll just be default to xor
	input_count = 2;
	pattern_count = 4;
	hidden_count = atoi(argv[2]);
	output_count = 1;
	eta = atof(argv[3]);
	epochs = atoi(argv[4]);
	memcpy(*xor_targets, *actual_target, 4 * sizeof(double));
    }
    else
    {
	fprintf(stderr, "neuralnet1 [#features] [#points] [#hidden] [#output] [eta] [epochs]\nneuralnet1 [dataset] [#hidden] [eta] [epochs]\n");
	return 1;
    }
    srand(time(NULL));

    struct Network* network = create_network(input_count, pattern_count,
	    hidden_count, output_count,
	    eta, epochs);
    memcpy(*inputs, *network->inputs, 8 * sizeof(double));
    memcpy(*actual_target, *network->targets, 4 * sizeof(double));
    //gen_random_data(pattern_count, input_count, &add);

    for (int e = 0; e < epochs; e++)
    {
	train_one_epoch(network);
	printf("%d,%f\n", e, network->error);
    }

    return 0;
}
