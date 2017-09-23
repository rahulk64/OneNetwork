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

int[] forward(int[] examples) {
  //TODO
}

int[] backward(int[] examples, int[] results) {
  //TODO
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