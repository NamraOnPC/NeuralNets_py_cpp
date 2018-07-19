//from numpy import exp, array, random, dot
#include <algorithm>  
#include <cmath>      
#include <cstdlib>    
#include <iomanip>    
#include <iostream>
#include <numeric>    
#include <vector>      
using namespace std;

void vectorPrint(vector<double>& V) {

	for (auto col : V)

		cout << col << " ";

	cout << "\n";

}

void matrixPrint(vector<vector<double>>& M) {

	for (auto row : M)

		vectorPrint(row);

}

auto matTranspose(vector<vector<double>>& Y, vector<vector<double>>& X) {

	size_t rows = X.size();    //  number of rows    for matrix X

	size_t cols = X[0].size(); //  number of columns for matrix X

	Y.resize(cols);             // set nunber of rows for Y

	for (auto&e : Y)             // set nunber of cols for each row of Y

		e.resize(rows);

	for (size_t r = 0; r < rows; r++)   // copy data

		for (size_t c = 0; c < cols; c++)

			Y[c][r] = X[r][c];

} 

void matMult(vector<double>& Y, vector<vector<double>>& M, vector<double>& X) { // Y = M * X

	for (size_t i = 0; i < M.size(); i++) {

		Y[i] = inner_product(M[i].begin(), M[i].end(), X.begin(), 0.);

	}

};





//class NeuralNetwork() :

class NeuralNetwork {
	
public:
	//def __init__(self) :
	//random.seed(1)
	//self.synaptic_weights = 2 * random.random((3, 1)) - 1

	vector<double> synaptic_weights; // bad practice ( but couldn't make it work if private)

	NeuralNetwork(size_t s) {

		synaptic_weights.resize(3);
		
		srand(1);

		for (auto& e : synaptic_weights)
			e = 2. * rand() / double(RAND_MAX) - 1.;


	}





	//def __sigmoid(self, x) :
	//return 1 / (1 + exp(-x))
	double __sigmoid(double x) { return 1. / (1. + exp(-x)); }
	void __sigmoid(vector<double>& output, vector<double> input) {

		transform(input.begin(), input.end(),
			output.begin(),
			[this](double i) { return __sigmoid(i); });

	}
	//def __sigmoid_derivative(self, x) :
	//return x * (1 - x)
	double __sigmoid_derivative(double x) { return x * (1. - x ); }
	

	//def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations) :
	//for iteration in xrange(number_of_training_iterations) :
		//output = self.think(training_set_inputs)
		//error = training_set_outputs - output
		//adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
		//self.synaptic_weights += adjustment
void train(vector<vector<double>>& training_set_inputs, vector<double>& training_set_outputs, int number_of_training_iterations) {

	vector<double> output(training_set_outputs.size());
	
	vector<double> error(training_set_outputs.size());

	vector<vector<double>> training_set_inputsT;

	matTranspose(training_set_inputsT, training_set_inputs);

	vector<double> adjustment (training_set_inputs[0].size());

	for (int iteration = 0; iteration < number_of_training_iterations; iteration++) {

		think(output, training_set_inputs);

		transform(training_set_outputs.begin(), training_set_outputs.end(),
			output.begin(),
			error.begin(),
			[](double t, double o) {return t - o; }); // or std::minus()


		transform(error.begin(), error.end(),
			output.begin(),
			error.begin(),
			[this](double e, double o) { return e * __sigmoid_derivative(o);  });

		matMult(adjustment, training_set_inputsT, error);

		transform(synaptic_weights.begin(), synaptic_weights.end(),
			adjustment.begin(),
			synaptic_weights.begin(),
			[](double w, double a) { return w + a; });




	}

	}



		//def think(self, inputs) :
		//return self.__sigmoid(dot(inputs, self.synaptic_weights))

void think(vector<double>& output , vector<vector<double>>& inputs) {

	matMult(output, inputs, synaptic_weights);

	__sigmoid(output, output);

	}

double think(vector<double>& input) {

	return __sigmoid(inner_product(input.begin(), input.end() , synaptic_weights.begin(), 0. ));

}





};
		//if __name__ == "__main__" :

int main() {
	//neural_network = NeuralNetwork()
	NeuralNetwork neural_network(3);

	//print "Random starting synaptic weights: "
	//print neural_network.synaptic_weights
	cout << "Random starting synaptic weights: ";
	vectorPrint(neural_network.synaptic_weights);


	//training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	//training_set_outputs = array([[0, 1, 1, 0]]).T
	vector<vector<double>> training_set_inputs = { { 0, 0, 1 },{ 1, 1, 1 },{ 1, 0, 1 },{ 0, 1, 1 } };
	vector<double> training_set_outputs = { 0, 1, 1, 0 };

	//neural_network.train(training_set_inputs, training_set_outputs, 10000)
	neural_network.train(training_set_inputs, training_set_outputs, 10000);

	//print "New synaptic weights after training: "
	//print neural_network.synaptic_weights
	cout << "New synaptic weights after training: ";
	vectorPrint(neural_network.synaptic_weights);

	//print "Considering new situation [1, 0, 0] -> ?: "
	//print neural_network.think(array([1, 0, 0]))
	cout << "Considering new situation [1, 0, 0] -> ?: ";
	
	vector<double> input = { 1, 0, 0 };

	cout << setprecision(8) << neural_network.think(input) << "\n";



}
