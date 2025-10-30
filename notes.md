# Neural Network (from scratch)

## Notes

**Cost Function**

* **cost = number of errors / number of total predictions**

* **Low cost = good model**, **high cost = bad model**

* **Goal:** Minimize cost metric as much as possible

**Gradient Descent**

* Assume we have a cost function $C$, which has $n$ variables, $w_1, w_2, \dots, w_{n}$

* Represent $C$ as $C(w_1, w_2, \dots, w_{n})$

* The gradient of $C$ is denoted as $\nabla C$, which is the collection of all partial derivatives of $C$ with respect to each one of its $n$ variables $$\nabla C = [\frac{\partial C}{\partial w_1}, \frac{\partial C}{\partial w_2}, \dots, \frac{\partial C}{\partial w_{n}}]$$

* The gradient allows us to dial in our cost function accordingly

**Perceptron Model**

* A Neural Network is modeled after a human neuron

* A neuron takes in an input, does something, and returns an output

* Networks can be represented as a collection of **nodes**

* Each node has a numeric value

* Nodes are connected together using **weights**

**Rules of a Network:**

* Each **node** in a **layer** connects to every other node in the next layer through **weights**
    
* Every **weight** has a predetermined value

* Only the first **layer of nodes** has initial values, referred to as the **input layer**

* Every other **layer of nodes** gets calculated based on the previous layer and the weights connecting them

* Given $i$ nodes in the first layer of a network, and $j$ nodes in the second layer, there will be $i \cdot j$ weights connecting them

**Biases**

* A **bias** is a constant term added to a node at the end of the dot product between a node and weight

* Applied to every node of each layer, excluding the input layer

* Bias is significant as it prevents the case of the weights being 0 affecting the overall result of the network

    * **e.g.** if each weight in the network was 0, the resulting neuron values for the following layer would always be 0. This value of 0 would propagate throughout each following layer, until the entire network would compute 0.

**Rules of Biases**

* Each node has its own randomly assigned bias value, except for nodes in the input layer

* Biases are added to the value of a neuron after the dot product calculation between the node and the weight

**Neural Network Intuition**

* Composed of **Weights, Nodes, Biases, and Cost**

* **Weights** and **Biases** are the dials that can be changed to get a desired output

* The first layer is excluded since it is the raw input we send into the network

* Neural Network computes a **cost function** $C$ that includes all of the **weights** and **biases**

* It will learn how to update the **weights** and **biases** accordingly to minimize the coszt through the **gradient** $\nabla C$

**Layers of a Network**

* **Input Layer:** 
    
    * First layer in a neural network
    
    * Raw input to the model

* **Hidden Layers:**

    * All layers in between input and output layer

    * The actions taken by the hidden layers are not interpretable

    * Can only interpret what the input layer takes in as raw input and what the output layers prediction means

* **Output Layer:**

    * Last layer in the network

    * Outputs the prediction of the network

**Process of Feed Forward Neural Network**

* Randomly initialize the weights and biases of the network

* Perform feed forward process for each training sample in the dataset

* Get $\hat{y}$ for each training sample at the end of the feed forward process

* Calculate cost metric for the network across all training samples

**Back Propagation**

* Calculate $\frac{\partial C}{\partial W^{[L]}}$ and $\frac{\partial C}{\partial b^{[L]}}$ for the final layer $L$

* Calculate the propagator for the penultimate layer $L - 1$ by finding $\frac{\partial C}{\partial A^{[L-1]}}$

* For all layers $l$ starting from $l = L - 1$, and going until the first layer $l = 1$, calculate $\frac{\partial C}{\partial W^{[l]}}, \frac{\partial C}{\partial b^{[l]}}$, and the propagator for the next layer $\frac{\partial C}{\partial A^{[l-1]}}$

## Summary

**Procedure of a Neural Network**

1. Provide input data to the network as the **input layer**,

2. The network then uses those values from the input layer and the weights connecing it to the second layer to compute the values of the second layer

3. This process propagates through all layers in the network, until each node in the network has a value

4. The last layer will output the networks prediction, which can then be compared to the labeled data to provide a cost metric for the network

5. Based on the cost, a gradient can be calculated with respect to $\nabla C$ and update the weights and biases accordingly

6. Repeat steps 1-5 until the cost is minimized as much as possible

**Rules of Neural Networks**

* For a node in a layer, its connected to every single node in the next layer through connections called weights

* Weights and biases are randomly initialized to begin with, but we change them based on the gradient $\nabla C$

* Every node in the network --except for nodes in the input layer-- have a bias, which is added to the node after the weight-node calculation.

---