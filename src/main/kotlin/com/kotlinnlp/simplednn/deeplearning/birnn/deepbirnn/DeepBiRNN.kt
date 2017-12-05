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
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The DeepBiRNN.
 *
 * A deep bidirectional RNN (or k-layer BiRNN) is composed of k BiRNN
 * BiRNN(1), ..., BiRNN(k) that feed into each other: the output of BIRNN(i)
 * becomes the input of BiRNN(i+1). Stacking BiRNNs in this way has been empirically
 * shown to be effective (Irsoy and Cardie, 2014).
 *
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property hiddenActivation the activation function of the hidden layer
 * @property recurrentConnectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 * @property numberOfLayers number of stacked BiRNNs
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 *
 * Note: each BiRNN has the hidden layer of the same size as of its input.
 */
class DeepBiRNN(
  val inputType: LayerType.Input,
  val inputSize: Int,
  val hiddenActivation: ActivationFunction?,
  val recurrentConnectionType: LayerType.Connection,
  val numberOfLayers: Int = 1,
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
  val layers: List<BiRNN> = this.buildBiRNNLayers()

  /**
   * The size of the output layer (of the last BiRNN)
   */
  val outputSize: Int = this.layers.last().outputSize

  /**
   * Serialize this [DeepBiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [DeepBiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   *
   */
  private fun buildBiRNNLayers(): List<BiRNN> {
    require(this.numberOfLayers > 0) { "required at least one BiRNN layer" }
    require(this.recurrentConnectionType.property == LayerType.Property.Recurrent) {
      "required recurrentConnectionType with Recurrent property"
    }

    var prevInputSize: Int = this.inputSize

    return List(
      size = this.numberOfLayers,
      init = { i ->

        val biRNN = this.buildBiRNN(
          inputSize = prevInputSize,
          inputType = if (i == 0) this.inputType else LayerType.Input.Dense,
          hiddenSize = prevInputSize)

        prevInputSize = biRNN.outputSize

        biRNN
      }
    )
  }

  /**
   * @param inputSize the size of the input arrays
   * @param inputSize the type of the input arrays
   * @param hiddenSize the output size of each RNN
   *
   * @return a new BiRNN
   */
  private fun buildBiRNN(inputSize: Int, inputType: LayerType.Input, hiddenSize: Int) = BiRNN(
    inputSize = inputSize,
    inputType = inputType,
    hiddenSize = hiddenSize,
    hiddenActivation = this.hiddenActivation,
    recurrentConnectionType = this.recurrentConnectionType,
    weightsInitializer = this.weightsInitializer,
    biasesInitializer = this.biasesInitializer)
}
