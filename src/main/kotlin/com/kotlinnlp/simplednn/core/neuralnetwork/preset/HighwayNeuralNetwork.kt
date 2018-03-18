package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * The Highway Neural Network factory.
 */
object HighwayNeuralNetwork {

  /**
   * @property inputSize the size of the input layer
   * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
   * @property inputDropout the dropout probability of the input (default 0.0).If applying it, the usual value is 0.25.
   * @property hiddenSize the size of the hidden layer
   * @property hiddenActivation the activation function of the hidden layer
   * @property hiddenDropout the dropout probability of the hidden (default 0.0).
   * @property hiddenMeProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
   * @property depth the number of hidden highway layers (default 1)
   * @property outputSize the size of the output layer
   * @property outputActivation the activation function of the output layer
   * @property outputMeProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
   * @property weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @property biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      inputDropout: Double = 0.0,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      hiddenDropout: Double = 0.0,
                      hiddenMeProp: Boolean = false,
                      depth: Int = 1,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      outputMeProp: Boolean = false,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()): NeuralNetwork {

    val layersConfiguration = ArrayList<LayerConfiguration>()

    layersConfiguration.add(LayerConfiguration(
      size = inputSize,
      inputType = inputType,
      dropout = inputDropout
    ))

    layersConfiguration.add(LayerConfiguration(
      size = hiddenSize,
      activationFunction = hiddenActivation,
      connectionType = LayerType.Connection.Feedforward,
      dropout = hiddenDropout,
      meProp = hiddenMeProp
    ))

    layersConfiguration.addAll((0 until depth).map {
      LayerConfiguration(
        size = hiddenSize,
        activationFunction = hiddenActivation,
        connectionType = LayerType.Connection.Highway,
        meProp = outputMeProp
      )
    })

    layersConfiguration.add(LayerConfiguration(
      size = outputSize,
      activationFunction = outputActivation,
      connectionType = LayerType.Connection.Feedforward,
      dropout = hiddenDropout,
      meProp = hiddenMeProp
    ))

    return NeuralNetwork(
      layerConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
  }
}
