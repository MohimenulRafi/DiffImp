# Supplementary Materials

## Model Architecture
We use the BERT-base model that has 12 layers of a transformer encoder block. Each block has 12 self-attention heads. It outputs a vector of hidden size 768. The BERT model has 110M parameters. We add a classifier layer on top of the BERT architecture for the classification task. The classification layer consists of one fully connected layer with dimension 128, followed by a rectifier linear unit (ReLU) activation, a dropout layer, and an output layer with one output neuron. The final output is passed through a sigmoid function to generate the probability.
