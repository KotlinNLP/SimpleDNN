package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 *
 */
object HighwayNeuralNetwork {

  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      inputDropout: Double = 0.0,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      hiddenDropout: Double = 0.0,
                      hiddenMeProp: Boolean = false,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      outputMeProp: Boolean = false,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()) = NeuralNetwork(
          LayerConfiguration(
                  size = inputSize,
                  inputType = inputType,
                  dropout = inputDropout
          ),
          LayerConfiguration(
                  size = hiddenSize,
                  activationFunction = hiddenActivation,
                  connectionType = LayerType.Connection.Highway,
                  dropout = hiddenDropout,
                  meProp = hiddenMeProp
          ),
          LayerConfiguration(
                  size = outputSize,
                  activationFunction = outputActivation,
                  connectionType = LayerType.Connection.Highway,
                  meProp = outputMeProp
          ),
          weightsInitializer = weightsInitializer,
          biasesInitializer = biasesInitializer
  )
}
