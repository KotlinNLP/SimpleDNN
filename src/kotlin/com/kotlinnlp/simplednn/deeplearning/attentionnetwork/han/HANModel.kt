/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN

/**
 * The model of the Hierarchical Attention Networks.
 *
 * @param inputSize the input size
 * @param hiddenActivation
 * @param hiddenConnectionType
 * @param outputActivation
 * @param attentionSize
 * @param outputSize
 * @param hierarchySize
 * @param compressionFactors
 *
 */
data class HANModel(
  val inputSize: Int,
  val hiddenActivation: ActivationFunction?,
  val hiddenConnectionType: LayerType.Connection,
  val outputActivation: ActivationFunction?,
  val attentionSize: Int,
  val outputSize: Int,
  val hierarchySize: Int = 2,
  val compressionFactors: ArrayList<Double>
) {

  /**
   * Check the compatibility of the arguments.
   */
  init {
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
      val inputSize: Int = this.getEncodingInputSize(i)

      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = inputSize,
        hiddenSize = this.getEncoderOutputSize(inputSize = inputSize, levelIndex = i),
        hiddenActivation = this.hiddenActivation,
        recurrentConnectionType = this.hiddenConnectionType)
    }
  )

  /**
   * An array of parameters, one for each AttentionNetwork which composes the HAN.
   */
  val attentionNetworksParams = Array(
    size = this.hierarchySize,
    init = { i ->

      AttentionNetworkParameters(
        inputSize = this.biRNNs[i].hiddenSize,
        attentionSize = this.attentionSize,
        sparseInput = false
      )
    }
  )

  /**
   * The parameters of the output feedforward layer.
   */
  val outputLayerParams = FeedforwardLayerParameters(
    inputSize = this.biRNNs.last().hiddenSize,
    outputSize = this.outputSize,
    sparseInput = false)

  /**
   * @param levelIndex the index of a level of the hierarchy
   *
   * @return the size of the input at the given [levelIndex]
   */
  private fun getEncodingInputSize(levelIndex: Int): Int {

    var inputSize: Int = this.inputSize

    for (i in 0 until levelIndex) {
      inputSize = this.getEncoderOutputSize(inputSize = inputSize, levelIndex = levelIndex)
    }

    return inputSize
  }

  /**
   * Get the size of the output of the encoder at the given [levelIndex], compressing the [inputSize] with the
   * corresponding compression factor.
   *
   * Since the output of the encoder is the concatenation of the outputs of 2 RNNs, the output size could be rounded to
   * the next odd integer.
   *
   * @param inputSize the size of the input at the given [levelIndex]
   * @param levelIndex the index of a level of the hierarchy
   *
   * @return the output size of the encoder at the given [levelIndex]
   */
  private fun getEncoderOutputSize(inputSize: Int, levelIndex: Int): Int {

    val compressedInputSize = Math.round(inputSize * this.compressionFactors[levelIndex]).toInt()

    return if (compressedInputSize % 2 == 0) compressedInputSize else compressedInputSize + 1
  }
}
