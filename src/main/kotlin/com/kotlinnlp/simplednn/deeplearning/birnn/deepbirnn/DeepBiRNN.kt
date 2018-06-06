/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The DeepBiRNN.
 *
 * A deep bidirectional RNN (or k-layer BiRNN) is composed of k BiRNNs that feed into each other: the output of a
 * BIRNN(i) becomes the input of BiRNN(i+1). Stacking BiRNNs in this way has been empirically shown to be effective
 * (Irsoy and Cardie, 2014).
 *
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property hiddenActivation the activation function of the hidden layer
 * @property dropout the probability of dropout (default 0.0)
 * @param numberOfLayers number of stacked BiRNNs
 * @param recurrentConnectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 * @param gainFactors an array with numberOfLayers elements, which defines the gain factor of the output size of
 *                    each layer in respect of its input. By default the first factor is 2.0, the others 1.0.
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 *
 * Note: each BiRNN has the hidden layer of the same size as of its input.
 */
class DeepBiRNN(
  val inputType: LayerType.Input,
  val inputSize: Int,
  val hiddenActivation: ActivationFunction?,
  val dropout: Double = 0.0,
  numberOfLayers: Int = 1,
  recurrentConnectionType: LayerType.Connection,
  private val gainFactors: List<Double> = List(size = numberOfLayers, init = { i -> if (i == 0) 2.0 else 1.0 }),
  private val weightsInitializer: Initializer? = GlorotInitializer(),
  private val biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [DeepBiRNN] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [DeepBiRNN]
     *
     * @return the [DeepBiRNN] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): DeepBiRNN = Serializer.deserialize(inputStream)
  }

  /**
   * Stacked BiRNNs
   */
  val layers: List<BiRNN> = this.buildBiRNNLayers(
    numberOfLayers = numberOfLayers,
    recurrentConnectionType = recurrentConnectionType)

  /**
   * The model parameters.
   */
  val model = DeepBiRNNParameters(paramsPerBiRNN = this.layers.map { it.model })

  /**
   * The size of the output layer (of the last BiRNN)
   */
  val outputSize: Int = this.layers.last().outputSize

  /**
   * Check the compatibility of the arguments.
   */
  init {
    require(numberOfLayers > 0) { "The number of layers must be >= 1" }
    require(this.gainFactors.size == numberOfLayers) {
      "The number of gain factors (%d) doesn't match the number of layers (%d)"
        .format(this.gainFactors.size, numberOfLayers)
    }
  }

  /**
   * Serialize this [DeepBiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [DeepBiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * Build the BiRNN layers.
   *
   * @param numberOfLayers the number of layers
   * @param recurrentConnectionType the type of recurrent layers connection
   *
   * @return a list of BiRNNs
   */
  private fun buildBiRNNLayers(numberOfLayers: Int, recurrentConnectionType: LayerType.Connection): List<BiRNN> {
    require(numberOfLayers > 0) { "required at least one BiRNN layer" }
    require(recurrentConnectionType.property == LayerType.Property.Recurrent) {
      "required recurrentConnectionType with Recurrent property"
    }

    var inputSize: Int = this.inputSize

    return List(
      size = numberOfLayers,
      init = { i ->

        val outputSize: Int = this.getBiRNNOutputSize(inputSize = inputSize, layerIndex = i)
        val biRNN = this.buildBiRNN(
          inputSize = inputSize,
          inputType = if (i == 0) this.inputType else LayerType.Input.Dense,
          hiddenSize = outputSize / 2,
          recurrentConnectionType = recurrentConnectionType)

        inputSize = outputSize

        biRNN
      }
    )
  }

  /**
   * @param inputSize the size of the input arrays
   * @param inputSize the type of the input arrays
   * @param hiddenSize the output size of each RNN
   * @param recurrentConnectionType the type of recurrent layers connection
   *
   * @return a new BiRNN
   */
  private fun buildBiRNN(inputSize: Int,
                         inputType: LayerType.Input,
                         hiddenSize: Int,
                         recurrentConnectionType: LayerType.Connection) = BiRNN(
    inputSize = inputSize,
    inputType = inputType,
    hiddenSize = hiddenSize,
    hiddenActivation = this.hiddenActivation,
    dropout = this.dropout,
    recurrentConnectionType = recurrentConnectionType,
    weightsInitializer = this.weightsInitializer,
    biasesInitializer = this.biasesInitializer)

  /**
   * Get the size of the output of the BiRNN at the given [layerIndex], combining its [inputSize] with the related gain
   * factor.
   *
   * Since the output of the BiRNN is the concatenation of the outputs of 2 RNNs, the output size must be rounded to
   * an odd integer (the next following in this case).
   *
   * @param inputSize the size of the input at the given [layerIndex]
   * @param layerIndex the index of a layer
   *
   * @return the output size of the BiRNN at the given [layerIndex]
   */
  private fun getBiRNNOutputSize(inputSize: Int, layerIndex: Int): Int {

    val gain: Double = this.gainFactors[layerIndex]
    val roughOutputSize = Math.round(gain * inputSize).toInt()

    return if (roughOutputSize % 2 == 0) roughOutputSize else roughOutputSize + 1
  }
}
