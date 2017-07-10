#pragma once

// Function Definitions
float** alloc_2d(size_t array_i, size_t array_j);
float sigmoid(float val);
float sig_prime(float val);

void init_values(float* array, size_t array_size);
void init_values2d(float** array, size_t array_i, size_t array_j);
void print_array(float* array, size_t array_size);
void print_array2d(float** array, size_t array_i, size_t array_j);

float* randomize_array(size_t array_size);

// Struct & Method Definitions
struct Network
{
    size_t input_count;
    size_t pattern_count;
    size_t hidden_count;
    size_t output_count;
    size_t epochs;

    float eta;

    float** inputs;
    float** targets;

    float* hidden_bias;
    float** hidden;
    
    float* output_bias;
    float** output;
    
    float** weights_ih;
    float** weights_ho;

    struct NetworkTrainer* trainer;
    float error;
};
struct NetworkTrainer
{
    float* hidden_delta; 
    float* hidden_bias_delta;
    float* sum_dow;
    
    float* output_delta;
    float* output_bias_delta;
    
    float** weights_ih_delta;
    float** weights_ho_delta;
};

struct Network* create_network(size_t input_count, size_t pattern_count,
	size_t hidden_count, size_t output_count,
	float learning_rate, size_t epochs);
struct NetworkTrainer* create_network_trainer(size_t input_count, size_t pattern_count,
	size_t hidden_count, size_t output_count);

void train_one_epoch(struct Network* network);
void feed_forward(struct Network* network, size_t p);
void backpropagate(struct Network* network, size_t p);
void update_weights(struct Network* network);
