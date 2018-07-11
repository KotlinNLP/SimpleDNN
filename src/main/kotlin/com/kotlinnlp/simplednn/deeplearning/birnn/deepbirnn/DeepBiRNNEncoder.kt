/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Deep Bidirectional Recursive Neural Network Encoder
 *
 * For convenience, this class exposes methods as if there was a single [BiRNN].
 * In this way, it is possible to use a [BiRNNEncoder] and a [DeepBiRNNEncoder] almost interchangeably.
 *
 * @property network the [DeepBiRNN] of this encoder
 * @property useDropout whether to apply the dropout during the [forward]
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property id an identification number useful to track a specific [DeepBiRNNEncoder]
 */
class DeepBiRNNEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val network: DeepBiRNN,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
): NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  DeepBiRNNParameters // ParamsType
  > {

  /**
   * List of encoders for all the stacked [BiRNN] layers.
   */
  private val encoders = this.network.levels.mapIndexed { i, biRNN ->
    if (i == 0)
      BiRNNEncoder<InputNDArrayType>(biRNN, useDropout = this.useDropout, propagateToInput = this.propagateToInput)
    else
      BiRNNEncoder<DenseNDArray>(biRNN, useDropout = this.useDropout, propagateToInput = true)
  }

  /**
   * The Forward.
   *
   * @param input the input sequence
   *
   * @return the result of the forward
   */
  override fun forward(input: List<InputNDArrayType>): List<DenseNDArray> {

    var output: List<DenseNDArray>

    @Suppress("UNCHECKED_CAST")
    output = (this.encoders[0] as BiRNNEncoder<InputNDArrayType>).forward(input)

    for (i in 1 until this.encoders.size) {
      @Suppress("UNCHECKED_CAST")
      output = (this.encoders[i] as BiRNNEncoder<DenseNDArray>).forward(output)
    }

    return output
  }

  /**
   * Propagate the errors of the entire sequence.
   *
   * @param outputErrors the errors to propagate
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    var errors: List<DenseNDArray> = outputErrors

    this.encoders.reversed().forEach { encoder ->
      encoder.backward(errors)
      errors = encoder.getInputErrors(copy = false)
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    this.encoders.first().getInputErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the DeepBiRNN parameters
   */
  override fun getParamsErrors(copy: Boolean) = DeepBiRNNParameters(
    paramsPerBiRNN = this.encoders.map { it.getParamsErrors(copy = copy) }
  )
}
