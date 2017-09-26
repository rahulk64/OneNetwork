#include "OneNetwork.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

OneNetwork::OneNetwork() {
  learningRate = 0.7;
  iterations = 100;
  hiddenNeurons = 3;
  inputNeurons = 2;
  outputNeurons = 1;

  setWeights();
}

OneNetwork::OneNetwork(double l, int i, int h) {
  learningRate = l;
  iterations = i;
  hiddenNeurons = h;
  inputNeurons = 2;
  outputNeurons = 1;

  setWeights();
}

OneNetwork::OneNetwork(int i, int h, int o) {
  learningRate = 0.7;
  iterations = 100;
  hiddenNeurons = h;
  inputNeurons = i;
  outputNeurons = o;

  setWeights();
}

double OneNetwork::getLearningRate() {
  return learningRate;
}

void OneNetwork::setLearningRate(double l) {
  learningRate = l;
}

int OneNetwork::getIterations() {
  return iterations;
}

void OneNetwork::setIterations(int i) {
  iterations = i;
}

int[] OneNetwork::getNeurons() {
  return {inputNeurons, hiddenNeurons, outputNeurons};
}

bool OneNetwork::setNeurons(int[] n) {
  int size = *(&n + 1) - n;
  if(size != 3) {
    return false;
  } else {
    inputNeurons = n[0];
    hiddenNeurons = n[1];
    outputNeurons = n[2];
  }

  setWeights();
}

void OneNetwork::learn(int[] examples) {
  for (int i = 0; i < iterations; i++) {
      int[] results = forward(examples);
      int[] errors = backward(examples, results);
      printArray(results, "results");
      printArray(errors, "errors");
    }
}

double sigmoid(double num) {
  return tanh(num);
}

double sigmoidPrime(double num) {
  double sech = 1.0 / cosh(x);
  return sech*sech; //tanh'(x) = sech^2(x)
}

void setWeights() {
  double weights1[inputNeurons][hiddenNeurons];
  double weights2[hiddenNeurons][outputNeurons];

  for(int i = 0; i < hiddenNeurons; i++) {
    for(int j = 0; j < inputNeurons; j++) {
      srand(time(NULL));
      weights1[j][i] = ((double) rand() / (RAND_MAX));
    }
  }

  for(int i = 0; i < outputNeurons; i++) {
    for(int j = 0; j < hiddenNeurons; j++) {
      srand(time(NULL));
      weights2[j][i] = ((double) rand() / (RAND_MAX));
    }
  }
}

double[] forward(int[] examples) {
  int size = *(&examples + 1) - examples;
  if(size != inputNeurons) {
    return {-1};
  }

  double hiddenArr[hiddenNeurons];
  double returnArr[outputNeurons];

  for(int i = 0; i < hiddenNeurons; i++) {
    double sum = 0;
    for(int j = 0; j < inputNeurons; j++) {
      sum += examples[j]*weights1[j][i];
    }
    hiddenArr[i] = sigmoid(sum);
  }

  for(int i = 0; i < outputNeurons; i++) {
    double sum = 0;
    for(int j = 0; j < hiddenNeurons; j++) {
      sum += hiddenArr[j]*weights2[j][i];
    }
    returnArr[i] = sigmoid(sum);
  }
}

double[] backward(int[] examples, double[] results, double[] expected) {
  int size = *(&results + 1) - results;
  if(size != outputNeurons) return {-1};

  //Adjust Weights between Hidden and Output Layers
  double marginError[size];
  for(int i = 0; i < size; i++) {
    marginError[i] = expected[i] - results[i];
  }

  double deltaSum[size];
  double hiddenArr[hiddenNeurons];
  double tempWeights2[hiddenNeurons][outputNeurons];

  for(int i = 0; i < hiddenNeurons; i++) {
    double sum = 0;
    for(int j = 0; j < inputNeurons; j++) {
      sum += examples[j]*weights1[j][i];
    }
    hiddenArr[i] = sigmoid(sum);
  }

  for(int i = 0; i < size; i++) {
    double sum = 0;
    for(int j = 0; j < hiddenNeurons; j++) {
      sum += hiddenArr[j]*weights2[j][i];
    }
    deltaSum[i] = sigmoidPrime(sum)*marginError[i];
  }

  for(int i = 0; i < outputNeurons; i++) {
    for(int j = 0; j < hiddenNeurons; j++) {
      tempWeights2[j][i] += deltaSum[i]/hiddenArr[j];
    }
  }

  //Adjust Weights between Input and Output Layers
  double deltaHidden[hiddenNeurons];
  double deltaHSum[hiddenNeurons];

  for(int i = 0; i < outputNeurons; i++) {
    for(int j = 0; j < hiddenNeurons; j++) {
      deltaHidden[j] = deltaSum[i]/weights2[j][i];
    }
  }

  for(int i = 0; i < hiddenNeurons; i++) {
    double sum = 0;
    for(int j = 0; j < inputNeurons; j++) {
      sum += examples[j]*weights1[j][i];
    }
    deltaHSum[i] = sigmoidPrime(sum)*deltaHidden[i];
  }

  for(int i = 0; i < hiddenNeurons; i++) {
    for(int j = 0; j < inputNeurons; j++) {
      weights1[j][i] = deltaHSum[i]/examples[j];
    }
  }

  weights2 = tempWeights2;

}

void printArray(int[] array, string s) {
  int size = *(&array + 1) - array;
  cout << s << ": ";
  for(int i = 0; i < size; i++) {
    cout << array[i] << ", ";
  }
  cout << endl;
}

void printNetwork() {
  //TODO
}
