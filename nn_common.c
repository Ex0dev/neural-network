#include <stdlib.h>
#include <math.h>

#include "nn_common.h"

double **alloc_2d(int array_i, int array_j)
{
    double **new_array = calloc(array_i * array_j, sizeof(double *));
    for (int i = 0; i < array_i; i++)
    {
        new_array[i] = calloc(array_j, sizeof(double));
    }
    return new_array;
}

double sigmoid(double val)
{
    return 1.0f / (1.0f + exp(-val));
}

double sig_prime(double val)
{
    return sigmoid(val) * (1.0f - sigmoid(val));
}

void init_values(double *array, int array_size)
{
    for (int i = 0; i < array_size; i++)
    {
        array[i] = 2.0 * (((double)rand() / (RAND_MAX + 1)) - 0.5) * 1;
    }
}

void init_values2d(double **array, int array_i, int array_j)
{
    for (int i = 0; i < array_i; i++)
    {
        init_values(array[i], array_j);
    }
}

void print_array(double *array, int array_size)
{
    printf("*********\n");
    for (int i = 0; i < array_size; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n*********\n");
}

void print_array2d(double **array, int array_i, int array_j)
{
    printf("*********\n");
    for (int i = 0; i < array_i; i++)
    {
        for (int j = 0; j < array_j; j++)
        {
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n*********\n");
}

double *randomize_array(int array_size)
{
    double *random_array = calloc(array_size, sizeof(double));
    for (int p = 0; p < array_size; p++)
    {
        random_array[p] = p;
    }
    for (int p = 0; p < array_size; p++)
    {
        int np = p + ((double)rand() / (RAND_MAX + 1)) * (array_size - p);
        int op = random_array[p];
        random_array[p] = np;
        random_array[np] = op;
    }
    return random_array;
}

struct Network* create_network(int input_count, int pattern_count,
	int hidden_count, int output_count,
	double learning_rate, int epochs)
{
    struct Network* network = malloc(sizeof(struct Network));
    if (network == NULL) return NULL;

    network->input_count = input_count;
    network->pattern_count = pattern_count;
    network->hidden_count = hidden_count;
    network->output_count = output_count;
    network->eta = learning_rate;
    network->epochs = epochs;

    network->inputs = alloc_2d(pattern_count, input_count);
    network->targets = alloc_2d(pattern_count, output_count);

    network->hidden_bias = calloc(hidden_count, sizeof(double));
    init_values(network->hidden_bias, hidden_count);
    network->hidden = alloc_2d(pattern_count, hidden_count);

    network->output_bias = calloc(hidden_count, sizeof(double));
    init_values(network->output_bias, output_count);
    network->output = alloc_2d(pattern_count, output_count);
    
    network->weights_ih = alloc_2d(input_count, hidden_count);
    network->weights_ho = alloc_2d(hidden_count, output_count);
    init_values2d(network->weights_ih, input_count, hidden_count);
    init_values2d(network->weights_ho, hidden_count, output_count);

    network->trainer = create_network_trainer(input_count, pattern_count,
	    hidden_count, output_count);
    network->error = 0.0f;

    return network;
}

struct NetworkTrainer* create_network_trainer(int input_count, int pattern_count,
	int hidden_count, int output_count)
{
    struct NetworkTrainer* trainer = malloc(sizeof(struct NetworkTrainer));
    if (trainer == NULL) return NULL;

    trainer->hidden_delta = calloc(hidden_count, sizeof(double));
    trainer->hidden_bias_delta = calloc(hidden_count, sizeof(double));
    trainer->sum_dow = calloc(hidden_count, sizeof(double));

    trainer->output_delta = calloc(output_count, sizeof(double));
    trainer->output_bias_delta = calloc(output_count, sizeof(double));

    trainer->weights_ih_delta = alloc_2d(input_count, hidden_count);
    trainer->weights_ho_delta = alloc_2d(hidden_count, output_count);

    return trainer;
}

void train_one_epoch(struct Network* network)
{
    network->error = 0.0f;

    int p, k;
    for (p = 0; p < network->pattern_count; p++)
    {
	feed_forward(network, p);
	backpropagate(network, p);

	for (k = 0; k < network->output_count; k++)
	{
	    network->error += pow(network->targets[p][k] - network->output[p][k], 2);
	}
    }
    network->error = network->error / (network->output_count * network->pattern_count);
    update_weights(network);
}

void feed_forward(struct Network* network, int p)
{
    int i, j, k;
    // Feed-forward to hidden neurons
    for (j = 0; j < network->hidden_count; j++)
    {
	network->hidden[p][j] = network->hidden_bias[j];
	for (i = 0; i < network->input_count; i++)
	{
	    network->hidden[p][j] += network->weights_ih[i][j]
		* network->inputs[p][i];
	}
	network->hidden[p][j] = sigmoid(network->hidden[p][j]);
    }

    // Feed-forward to output neurons
    for (k = 0; k < network->output_count; k++)
    {
	network->output[p][k] = network->output_bias[k];
	for (j = 0; j < network->hidden_count; j++)
	{
	    network->output[p][k] += network->weights_ho[j][k]
		* network->hidden[p][j];
	}
	network->output[p][k] = sigmoid(network->output[p][k]);
    }
}

void backpropagate(struct Network* network, int p)
{
    struct NetworkTrainer* trainer = network->trainer;
    int i, j, k;
    // Calculate output delta & change to hidden-output weights
    for (k = 0; k < network->output_count; k++)
    {
	trainer->output_delta[k] = (network->targets[p][k] - network->output[p][k]) * network->output[p][k] * (1.0f - network->output[p][k]);

	trainer->output_bias_delta[k] += network->eta * trainer->output_delta[k];
	for (j = 0; j < network->hidden_count; j++)
	{
	    trainer->weights_ho_delta[j][k] += network->eta * network->hidden[p][j] * trainer->output_delta[k];
	}
    }

    // Calculate hidden delta & change to input-hidden weights
    for (j = 0; j < network->hidden_count; j++)
    {
	for (k = 0; k < network->output_count; k++)
	{
	    trainer->sum_dow[j] += network->weights_ho[j][k] * trainer->output_delta[k];
	}
	trainer->hidden_delta[j] = network->hidden[p][j] * (1.0f - network->hidden[p][j]) * trainer->sum_dow[j];

	trainer->hidden_bias_delta[j] += network->eta * trainer->hidden_delta[j];
	for (i = 0; i < network->input_count; i++)
	{
	    trainer->weights_ih_delta[i][j] += network->eta * network->inputs[p][i] * trainer->hidden_delta[j];
	}
    }
}

void update_weights(struct Network* network)
{
    struct NetworkTrainer* trainer = network->trainer;
    int i, j, k;
    // Updates the hidden bias & input-hidden weights
    for (i = 0; i < network->input_count; i++)
    {
	for (j = 0; j < network->hidden_count; j++)
	{
	    network->hidden_bias[j] += trainer->hidden_bias_delta[j];
	    network->weights_ih[i][j] += trainer->weights_ih_delta[i][j];
	    trainer->weights_ih_delta[i][j] = 0.0f;
	    trainer->hidden_bias_delta[j] = 0.0f;
	}
    }

    // Updates the output bias & hidden-output weights
    for (j = 0; j < network->hidden_count; j++)
    {
	for (k = 0; k < network->output_count; k++)
	{
	    network->output_bias[k] += trainer->output_bias_delta[k];
	    network->weights_ho[j][k] += trainer->weights_ho_delta[j][k];
	    trainer->weights_ho_delta[j][k] = 0.0f;
	    trainer->output_bias_delta[k] = 0.0f;
	}
    }
}
