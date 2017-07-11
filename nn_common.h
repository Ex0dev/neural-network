#pragma once

// Function Definitions
double** alloc_2d(int array_i, int array_j);
double sigmoid(double val);
double sig_prime(double val);

void init_values(double* array, int array_size);
void init_values2d(double** array, int array_i, int array_j);
void print_array(double* array, int array_size);
void print_array2d(double** array, int array_i, int array_j);

double* randomize_array(int array_size);

// Struct & Method Definitions
struct Network
{
    int input_count;
    int pattern_count;
    int hidden_count;
    int output_count;
    int epochs;

    double eta;

    double** inputs;
    double** targets;

    double* hidden_bias;
    double** hidden;
    
    double* output_bias;
    double** output;
    
    double** weights_ih;
    double** weights_ho;

    struct NetworkTrainer* trainer;
    double error;
};
struct NetworkTrainer
{
    double* hidden_delta; 
    double* hidden_bias_delta;
    double* sum_dow;
    
    double* output_delta;
    double* output_bias_delta;
    
    double** weights_ih_delta;
    double** weights_ho_delta;
};

struct Network* create_network(int input_count, int pattern_count,
	int hidden_count, int output_count,
	double learning_rate, int epochs);
struct NetworkTrainer* create_network_trainer(int input_count, int pattern_count,
	int hidden_count, int output_count);

void train_one_epoch(struct Network* network);
void feed_forward(struct Network* network, int p);
void backpropagate(struct Network* network, int p);
void update_weights(struct Network* network);
