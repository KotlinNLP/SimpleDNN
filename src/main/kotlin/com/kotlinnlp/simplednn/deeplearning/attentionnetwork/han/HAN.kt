/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of the Hierarchical Attention Networks.
 *
 * @property hierarchySize the number of levels of the hierarchy
 * @property inputSize the size of each array of input
 * @property biRNNsActivation the activation function of the BiRNNs
 * @property biRNNsConnectionType the layer connection type of the BiRNNs
 * @property attentionSize the size of the attention arrays of the AttentionLayers
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer
 * @property dropout the probability of dropout (default 0.0). If applying it, the usual value is 0.25.
 * @property compressionFactors an array with [hierarchySize] elements, which defines the compression factor of the
 *                              input size of each hierarchical level in respect of its output, starting from the lowest
 *                              level. By default the first factor is 2.0, the others 1.0.
 */
class HAN(
  val hierarchySize: Int = 2,
  val inputSize: Int,
  val biRNNsActivation: ActivationFunction?,
  val biRNNsConnectionType: LayerType.Connection,
  val attentionSize: Int,
  val outputSize: Int,
  val outputActivation: ActivationFunction?,
  val dropout: Double = 0.0,
  val compressionFactors: Array<Double> = Array(
    size = hierarchySize,
    init = { i -> if (i == 0) 2.0 else 1.0 })
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [HAN] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [HAN]
     *
     * @return the [HAN] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): HAN = Serializer.deserialize(inputStream)
  }

  /**
   * Check the compatibility of the arguments.
   */
  init {
    require(this.hierarchySize > 0) { "The number of hierarchical levels must be >= 1" }
    require(this.compressionFactors.size == hierarchySize) {
      "The number of compression factors (%d) doesn't match the number of levels of the hierarchy (%d)"
        .format(this.compressionFactors.size, this.hierarchySize)
    }
  }

  /**
   * An array of [BiRNN]s, one for each level of the HAN.
   */
  val biRNNs = Array(
    size = this.hierarchySize,
    init = { i ->
      val inputSize: Int = this.getLevelInputSize(i)

      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = inputSize,
        hiddenSize = this.getBiRNNOutputSize(inputSize = inputSize, levelIndex = i) / 2,
        hiddenActivation = this.biRNNsActivation,
        dropout = this.dropout,
        recurrentConnectionType = this.biRNNsConnectionType)
    }
  )

  /**
   * An array of parameters, one for each AttentionNetwork which composes the HAN.
   */
  val attentionNetworksParams = Array(
    size = this.hierarchySize,
    init = { i ->

      AttentionNetworkParameters(
        inputSize = this.biRNNs[i].outputSize,
        attentionSize = this.attentionSize,
        sparseInput = false
      )
    }
  )

  /**
   * The network (a single Feedforward layer) which transforms the output of the last hierarchical level to the output
   * of the [HANEncoder].
   */
  val outputNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.biRNNs.first().outputSize, // level 0 = top level
      inputType = LayerType.Input.Dense
    ),
    LayerConfiguration(
      size = this.outputSize,
      activationFunction = this.outputActivation,
      connectionType = LayerType.Connection.Feedforward
    )
  )

  /**
   * All the parameters of this [HAN] model.
   */
  val params = HANParameters(
    biRNNs = Array(size = this.biRNNs.size, init = { i -> this.biRNNs[i].model }),
    attentionNetworks = Array(size = this.attentionNetworksParams.size, init = { i -> this.attentionNetworksParams[i]}),
    outputNetwork = this.outputNetwork.model
  )

  /**
   * Initialize the parameters of the sub-networks using the given random generator and value for the biases.
   *
   * @param randomGenerator a [RandomGenerator] (default: fixed range with radius 0.08)
   * @param biasesInitValue the init value for all the biases (default: 0.0)
   *
   * @return this [HAN]
   */
  fun initialize(randomGenerator: RandomGenerator = FixedRangeRandom(radius = 0.08, enablePseudoRandom = true),
                 biasesInitValue: Double = 0.0): HAN {

    this.biRNNs.forEach { it.initialize(randomGenerator = randomGenerator, biasesInitValue = biasesInitValue) }
    this.attentionNetworksParams.forEach {
      it.initialize(randomGenerator = randomGenerator, biasesInitValue = biasesInitValue)
    }
    this.outputNetwork.initialize(randomGenerator = randomGenerator, biasesInitValue = biasesInitValue)

    return this
  }

  /**
   * Serialize this [HAN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [HAN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @param levelIndex the index of a level of the hierarchy
   *
   * @return the input size of the hierarchical level at the given [levelIndex]
   */
  private fun getLevelInputSize(levelIndex: Int): Int {

    var inputSize: Int = this.inputSize // assign the size of the lowest level

    ((levelIndex + 1) until this.hierarchySize).reversed().forEach { i -> // skip lowest level (already assigned)
      inputSize = this.getBiRNNOutputSize(inputSize = inputSize, levelIndex = i)
    }

    return inputSize
  }

  /**
   * Get the size of the output of the BiRNN at the given [levelIndex], compressing the [inputSize] with the
   * corresponding compression factor.
   *
   * Since the output of the BiRNN is the concatenation of the outputs of 2 RNNs, the output size could be rounded to
   * the next odd integer.
   *
   * @param inputSize the size of the input at the given [levelIndex]
   * @param levelIndex the index of a level of the hierarchy
   *
   * @return the output size of the encoder at the given [levelIndex]
   */
  private fun getBiRNNOutputSize(inputSize: Int, levelIndex: Int): Int {

    val compressedInputSize = Math.round(inputSize * this.compressionFactors.reversed()[levelIndex]).toInt()

    return if (compressedInputSize % 2 == 0) compressedInputSize else compressedInputSize + 1
  }
}
