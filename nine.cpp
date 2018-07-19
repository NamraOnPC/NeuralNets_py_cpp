// nine.cpp: Convert the neural net in nine lines of python to C++
// https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

// Embed the python code in the C++ code as a C++ comment to document what the C++ code is doing

// Why not grep the embedded python code, strip off the C++ comment, and run it?
//   grep //PY nine.cpp | sed 's=.*//PY ==' | grep -v '==' | python 

//PY from numpy import exp, array, random, dot
#include <algorithm>   // transform
#include <cmath>       // exp
#include <cstdlib>     // srand, rand
#include <iomanip>     // setprecision
#include <iostream>
#include <numeric>     // inner_product
#include <vector>      // vectors vector<->, and matrices vector<vector<->>
using namespace std;

int main(int argc, char*argv[])
{
  auto vectorPrint = [] (vector<double>& V) {
      for(auto col : V)
        cout << col << " ";
      cout << "\n";
  };
  auto matrixPrint = [vectorPrint] (vector<vector<double>>& M) {
    for(auto row : M)
      vectorPrint(row);
  };
  auto matTranspose = [] (vector<vector<double>>& Y, vector<vector<double>>& X) {
    size_t rows =  X.size();    //  number of rows    for matrix X
    size_t cols =  X[0].size(); //  number of columns for matrix X
    Y.resize(cols);             // set nunber of rows for Y
    for(auto&e : Y)             // set nunber of cols for each row of Y
      e.resize(rows);
    for(size_t r = 0; r < rows; r++)   // copy data
      for(size_t c = 0; c < cols; c++)
        Y[c][r] = X[r][c];
  };

  //PY training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
  vector<vector<double>> training_set_inputs = {{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}};  // MATRIX
  cout << "training_set_inputs\n";
  matrixPrint(training_set_inputs);

  //PY training_set_outputs = array([[0, 1, 1, 0]]).T
  vector<double> training_set_outputs = {0, 1, 1, 0};                                         // VECTOR
  cout << "training_set_outputs: "; 
  vectorPrint(training_set_outputs);

  //PY random.seed(1);
  srand(1);

  //PY synaptic_weights = 2 * random.random((3, 1)) - 1
  // 2 x (a random number between 0 and 1) -1 gives us a random number between -1 and 1
  vector<double> synaptic_weights(training_set_inputs[0].size());
  for(auto& e : synaptic_weights)
            e = 2. * rand() / double(RAND_MAX) - 1.;

  cout << "synaptic_weights: "; 
  vectorPrint(synaptic_weights);

  vector<vector<double>> training_set_inputsT;
  matTranspose( training_set_inputsT, training_set_inputs);

  vector<double> output    (training_set_outputs.size());
  vector<double> correction(training_set_outputs.size());
  auto matMult = [] (vector<double>& Y, vector<vector<double>>& M, vector<double>& X) { // Y = M * X
    for(size_t i = 0; i < M.size(); i++)
      Y[i] = inner_product(M[i].begin(), M[i].end(), X.begin(), 0.);
  };

  //PY for iteration in xrange(10000):
  int iterate = 10000;
  cout << "iterate=" << iterate << "\n";
  for (int iteration = 0; iteration < iterate; iteration++) {

      //PY   output = 1 / (1 + exp(-dot(training_set_inputs, synaptic_weights)))  
      // prediction = sigmoid(input * weights)
      matMult(output, training_set_inputs, synaptic_weights);               // input * weights
      transform(output.begin(), output.end(),                         
                output.begin(),
                [] (double element) { return 1./ (1. + exp(-element)); });  // sigmoid

      //PY   synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
      // output = prediction
      // prediction error = training output - prediction
      // sigmoid derivative ( prediction ) = prediction*(1-prediction) 
      // correction = prediction error * derivative
      transform(training_set_outputs.begin(), training_set_outputs.end(),   // prediction error * sigmoid derivative
                output.begin(), 
                output.begin(),
                [] (double t, double o) { return (t - o) * o * (1. - o); } 
      );
      matMult(correction, training_set_inputsT, output);                    // weight correction
      transform(synaptic_weights.begin(), synaptic_weights.end(),           // update weights
                correction.begin(),
                synaptic_weights.begin(),
                [] (double weight, double correction) { return weight += correction; }
                );
  }
  cout << "synaptic_weights: ";
  vectorPrint( synaptic_weights );

  //PY print 1 / (1 + exp(-dot(array([1, 0, 0]), synaptic_weights)))
  vector<double> input ={1,0,0};
  cout << "input: "; 
  vectorPrint(input);
  cout << "1 / (1 + exp(-dot(input, synaptic_weights))): "; // sigmoid( dot(input, weights) )
  cout << setprecision(8)  // C++ defaults to 6 decimal places. Python  defaults to 8.
       << 1. / (1. + exp(-inner_product(input.begin(), input.end(), synaptic_weights.begin(), 0.) ) ) << "\n";
} 

/* py:
[ 0.99993704]
*/

/* c++:
training_set_inputs
0 0 1
1 1 1
1 0 1
0 1 1
training_set_outputs: 0 1 1 0
synaptic_weights: 0.680375 -0.211234 0.566198
iterate=10000
synaptic_weights: 9.67301 -0.208724 -4.62912
input: 1 0 0
1 / (1 + exp(-dot(input, synaptic_weights))): 0.999937
*/
