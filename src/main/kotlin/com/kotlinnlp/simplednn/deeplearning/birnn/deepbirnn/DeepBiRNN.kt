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
import com.kotlinnlp.simplednn.deeplearning.birnn.mergeconfig.ConcatMerge
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The DeepBiRNN.
 *
 * A deep bidirectional RNN (or k-layer BiRNN) is composed of k BiRNNs that feed into each other: the output of the i-th
 * BIRNN becomes the input of the (i+1)-th BiRNN. Stacking BiRNNs in this way has been empirically shown to be effective
 * (Irsoy and Cardie, 2014).
 *
 * The output size of each BiRNN must be equal to the input size of the following one.
 *
 * @property levels the list of BiRNNs
 */
class DeepBiRNN(val levels: List<BiRNN>) : Serializable {

  /**
   * @property levels the list of BiRNNs
   */
  constructor(vararg levels: BiRNN): this(levels.toList())

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
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

    /**
     * Build a DeepBiRNN stacking more BiRNNs.
     *
     * Each BiRNN reduces (or increases) the size of its output respect to its input thanks to its hidden layer size
     * and a concatenation of the output of its two RNNs.
     * The gain factor between the input and the output of each BiRNN is controlled passing a list of gain factors, one
     * for each level.
     *
     * @param inputSize the input size
     * @param inputType the input type
     * @param recurrentConnectionType the type of recurrent layers connection
     * @param dropout the dropout of the recurrent layers
     * @param numberOfLevels the number of BiRNN levels
     * @param gainFactors the gain factors between the input size and the output size of each BiRNN
     * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
     * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
     *
     * @return a list of BiRNNs
     */
    fun byReduceConcat(inputSize: Int,
                       inputType: LayerType.Input,
                       recurrentConnectionType: LayerType.Connection,
                       hiddenActivation: ActivationFunction?,
                       dropout: Double = 0.0,
                       numberOfLevels: Int,
                       gainFactors: List<Double> = List(
                         size = numberOfLevels,
                         init = { i -> if (i == 0) 2.0 else 1.0 }),
                       weightsInitializer: Initializer? = GlorotInitializer(),
                       biasesInitializer: Initializer? = null): DeepBiRNN {

      require(numberOfLevels > 0) { "Required at least one BiRNN level." }

      require(recurrentConnectionType.property == LayerType.Property.Recurrent) {
        "Required a recurrent connection type with Recurrent property."
      }

      require(gainFactors.size == numberOfLevels) {
        "The number of gain factors (%d) doesn't match the number of layers (%d)"
          .format(gainFactors.size, numberOfLevels)
      }

      var levelInputSize: Int = inputSize

      return DeepBiRNN(List(
        size = numberOfLevels,
        init = { i ->

          val outputSize: Int = this.getBiRNNOutputSize(inputSize = levelInputSize, gain = gainFactors[i])

          val biRNN = BiRNN(
            inputSize = levelInputSize,
            inputType = if (i == 0) inputType else LayerType.Input.Dense,
            hiddenSize = outputSize / 2,
            hiddenActivation = hiddenActivation,
            dropout = dropout,
            recurrentConnectionType = recurrentConnectionType,
            outputMergeConfiguration = ConcatMerge(),
            weightsInitializer = weightsInitializer,
            biasesInitializer = biasesInitializer)

          levelInputSize = outputSize

          biRNN
        }
      ))
    }

    /**
     * Get the size of the output of a BiRNN level.
     *
     * Since the output of the BiRNN which uses a ConcatMerge is the concatenation of the outputs of 2 RNNs, the output
     * size must be rounded to an odd integer (the next following in this case).
     *
     * @param inputSize the size of the input
     * @param gain the gain factor to calculate the output size
     *
     * @return the output size of a BiRNN level
     */
    private fun getBiRNNOutputSize(inputSize: Int, gain: Double): Int {

      val roughOutputSize = Math.round(gain * inputSize).toInt()

      return if (roughOutputSize % 2 == 0) roughOutputSize else roughOutputSize + 1
    }
  }

  /**
   * The model parameters.
   */
  val model = DeepBiRNNParameters(paramsPerBiRNN = this.levels.map { it.model })

  /**
   * The size of the input level (the first BiRNN).
   */
  val inputSize: Int = this.levels.first().inputSize

  /**
   * The size of the output level (the last BiRNN).
   */
  val outputSize: Int = this.levels.last().outputSize

  /**
   * Check the compatibility of the BiRNNs.
   */
  init {
    require(this.levels.isNotEmpty()) { "The list of BiRNNs cannot be empty." }
    require(this.areLevelsCompatible()) { "The input-output size of the levels must be compatible." }
  }

  /**
   * Serialize this [DeepBiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [DeepBiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @return a boolean indicating if the input-output size of the levels are compatible
   */
  private fun areLevelsCompatible(): Boolean {

    var lastOutputSize: Int = this.levels.first().inputSize

    return this.levels.all {

      val isCompatible: Boolean = it.inputSize == lastOutputSize

      lastOutputSize = it.outputSize

      isCompatible
    }
  }
}
