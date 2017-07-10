// First attempt at making a feed-forward back-propogating neural network in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nn_common.h"

float inputs[4][2] = {0, 0,  0, 1,  1, 0,  1, 1};
float and_targets[4][1] = {0, 0, 0, 1};
float xor_targets[4][1] = {0, 1, 1, 0};
float or_targets[4][1] = {0, 1, 1, 1};
float nand_targets[4][1] = {1, 1, 1, 0};
float nor_targets[4][1] = {1, 0, 0, 0};

int main(int argc, char **argv)
{
    if (argc != 7)
    {
	fprintf(stderr, "neuralnet1 [#features] [#points] [#hidden] [#output] [eta] [epochs]");
	return 1;
    }
    srand(time(NULL));

    int input_count = atoi(argv[1]);
    int pattern_count = atoi(argv[2]);
    int hidden_count = atoi(argv[3]);
    int output_count = atoi(argv[4]);
    float eta = atof(argv[5]);
    int epochs = atoi(argv[6]);

    struct Network* network = create_network(input_count, pattern_count,
	    hidden_count, output_count,
	    eta, epochs);
    memcpy(*xor_inputs, *network->inputs, 8 * sizeof(float));
    memcpy(*xor_targets, *network->targets, 4 * sizeof(float));
    //gen_random_data(pattern_count, input_count, &add);

    for (size_t e = 0; e < epochs; e++)
    {
	train_one_epoch(network);
	printf("Epoch %zd Mean Squared Error: %f\n", e, network->error);
    }

    return 0;
}
