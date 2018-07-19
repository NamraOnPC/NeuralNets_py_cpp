#ifndef NEURON_H
#define NEURON_H
#include <algorithm>  
#include <cmath>      
#include <cstdlib>    
#include <iomanip>    
#include <iostream>
#include <numeric>    
#include <vector>      
using namespace std;


class NeuralNetwork {

private:

	int number_of_training_iterations;

	vector<vector<double>> training_set_inputs = { { 0, 0, 1 },{ 1, 1, 1 },{ 1, 0, 1 },{ 0, 1, 1 } };

	vector<double> training_set_outputs = { 0, 1, 1, 0 };

	vector<vector<double>> training_set_inputsT;

	vector<double> output;

	vector<double> correction;

public:

	vector<double> synaptic_weights;

	void train(vector<vector<double>> training_set_inputs, vector<double> training_set_outputs, int number_of_training_iterations);

	vector<double> matMult(vector<double>& Y, vector<vector<double>>& M, vector<double>& X);
};



#endif
