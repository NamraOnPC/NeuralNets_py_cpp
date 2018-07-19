
#include "Neuron.h"

int main(int argc, char*argv[]) {
	
	auto vectorPrint = [](vector<double>& V) {
		
		for (auto col : V)
		
			cout << col << " ";
		
		cout << "\n";
	
	};

	auto matrixPrint = [vectorPrint](vector<vector<double>>& M) {
		
		for (auto row : M)
		
			vectorPrint(row);
	
	};

	auto matTranspose = [](vector<vector<double>>& Y, vector<vector<double>>& X) {
		
		size_t rows = X.size();    //  number of rows    for matrix X
		
		size_t cols = X[0].size(); //  number of columns for matrix X
		
		Y.resize(cols);             // set nunber of rows for Y
		
		for (auto&e : Y)             // set nunber of cols for each row of Y
		
			e.resize(rows);
		
		for (size_t r = 0; r < rows; r++)   // copy data
		
			for (size_t c = 0; c < cols; c++)
			
				Y[c][r] = X[r][c];
	
	}; 

	auto matMult = [](vector<double>& Y, vector<vector<double>>& M, vector<double>& X) { // Y = M * X

		for (size_t i = 0; i < M.size(); i++)

			Y[i] = inner_product(M[i].begin(), M[i].end(), X.begin(), 0.);

	};


	vector<vector<double>> training_set_inputs = { { 0, 0, 1 },{ 1, 1, 1 },{ 1, 0, 1 },{ 0, 1, 1 } };  

	vector<double> training_set_outputs = { 0, 1, 1, 0 };                                        


	NeuralNetwork neural_network;

	srand(1);

	for (auto& e : neural_network.synaptic_weights)
		e = 2. * rand() / double(RAND_MAX) - 1.;

	cout << "Random starting synaptic weights : " << endl;

	vectorPrint(neural_network.synaptic_weights);

	vector<vector<double>> training_set_inputsT;
	matTranspose(training_set_inputsT, training_set_inputs);

	vector<double> output(training_set_outputs.size());
	vector<double> correction(training_set_outputs.size());

	neural_network.train(training_set_inputs, training_set_outputs, 10000);

	cout << "New synaptic weights after training: " << endl;
	vectorPrint(neural_network.synaptic_weights);

	cout << "Considering new situation [1, 0, 0] -> ?: " << endl;

	vector<double> input = { 1,0,0 };

	cout << setprecision(8) << 1. / (1. + exp(-inner_product(input.begin(), input.end(), neural_network.synaptic_weights.begin(), 0.))) << "\n";

}



void NeuralNetwork::train(vector<vector<double>> training_set_inputs, vector<double> training_set_outputs, int number_of_training_iterations) {

	for (int iteration = 0; iteration < number_of_training_iterations; iteration++) {


		matMult(this->output, training_set_inputs, synaptic_weights);

		transform(output.begin(), output.end(),
			
			output.begin(),
			
			[](double element) { return 1. / (1. + exp(-element)); }); 


		transform(training_set_outputs.begin(), training_set_outputs.end(), 
			
			output.begin(),
			
			output.begin(),
			
			[](double t, double o) { return (t - o) * o * (1. - o); }
		
		);

		matMult(correction, training_set_inputsT, output);                    
		
		transform(synaptic_weights.begin(), synaptic_weights.end(),         
		
			correction.begin(),
			
			synaptic_weights.begin(),
			
			[](double weight, double correction) { return weight += correction; }
		
		);

		

	}

}

vector<double> NeuralNetwork::matMult(vector<double>& Y, vector<vector<double>>& M, vector<double>& X) { // Y = M * X

	for (size_t i = 0; i < M.size(); i++) {

		Y[i] = inner_product(M[i].begin(), M[i].end(), X.begin(), 0.);

		return Y;

	}

};
