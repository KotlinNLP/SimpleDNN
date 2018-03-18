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
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense, Sparse, SparseBinary)
   * @param inputDropout the dropout probability of the input (default 0.0). If applying it, the usual value is 0.25.
   * @param hiddenSize the size of the hidden layer
   * @param hiddenActivation the activation function of the hidden layer
   * @param hiddenDropout the dropout probability of the hidden (default 0.0).
   * @param hiddenMeProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
   * @param numOfHighway the number of hidden highway layers (at least 1, the default)
   * @param outputSize the size of the output layer
   * @param outputActivation the activation function of the output layer
   * @param outputMeProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      inputDropout: Double = 0.0,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      hiddenDropout: Double = 0.0,
                      hiddenMeProp: Boolean = false,
                      numOfHighway: Int = 1,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      outputMeProp: Boolean = false,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()): NeuralNetwork {

    require(numOfHighway >= 1) { "The number of highway layers must be >= 1." }

    val layersConfiguration = mutableListOf<LayerConfiguration>()

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

    layersConfiguration.addAll((0 until numOfHighway).map {
      LayerConfiguration(
        size = hiddenSize,
        activationFunction = hiddenActivation,
        connectionType = LayerType.Connection.Highway,
        dropout = hiddenDropout,
        meProp = hiddenMeProp
      )
    })

    layersConfiguration.add(LayerConfiguration(
      size = outputSize,
      activationFunction = outputActivation,
      connectionType = LayerType.Connection.Feedforward,
      meProp = outputMeProp
    ))

    return NeuralNetwork(
      layerConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
  }
}
