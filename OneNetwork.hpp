#ifndef ONE_NETWORK
#define ONE_NETWORK

class OneNetwork {
public:
  OneNework(); //Default Constructor
  OneNetwork(double l, int i, int h); //Takes in learning rate, #iterations, and #hidden neurons
  OneNetwork(int i, int h, int o); //Takes in #input, hidden, & output neurons

  double getLearningRate();
  void setLearningRate(double l);

  int getIterations();
  void setIterations(int i);

  int[] getNeurons();
  void setNeurons(int[] n);

  void learn(int[] examples);
  void printNetwork();

private:
  double learningRate;
  int iterations;
  int hiddenNeurons, inputNeurons, outputNeurons;
  double[][] weights1, weight2;


  double sigmoid(double num);
  double sigmoidPrime(double num);
  void setWeights();
  double forward(int[] examples);
  double backward(int[] examples, int[] expected);
  void printArray(int[] array);
};

#endif
