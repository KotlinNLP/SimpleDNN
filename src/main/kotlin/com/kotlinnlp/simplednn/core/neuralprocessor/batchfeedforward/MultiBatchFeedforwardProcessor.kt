/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A Feed-forward processor with multiple parallel outputs for each input.
 * It encodes a sequence of arrays into another sequence of n parallel arrays using more networks.
 *
 * @property model list of feed-forward network models
 * @property useDropout whether to apply the dropout during the forward
 * @property propagateToInput whether to propagate the errors to the input during the backward
 */
class MultiBatchFeedforwardProcessor<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: List<StackedLayersParameters>,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<List<DenseNDArray>>, // OutputType
  List<List<DenseNDArray>>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * A list of processors which encode each input array into multiple vectors.
   */
  private val encoders: List<BatchFeedforwardProcessor<InputNDArrayType>> =
    this.model.map { BatchFeedforwardProcessor<InputNDArrayType>(
      model = it,
      useDropout = this.useDropout,
      propagateToInput = this.propagateToInput
    ) }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a
   *             reference (ignored, the value is always copied)
   *
   * @return the errors of the input sequence
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {

    val inputErrors: List<DenseNDArray> = this.encoders[0].getInputErrors(copy = true)

    for (encoderIndex in 1 until (this.model.size - 1)) {
      inputErrors.zip(this.encoders[encoderIndex].getInputErrors(copy = false)).forEach {
        (baseErrors, errors) -> baseErrors.assignSum(errors)
      }
    }

    return inputErrors
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the parameters errors of the sub-networks
   */
  override fun getParamsErrors(copy: Boolean) = this.encoders.flatMap { it.getParamsErrors(copy = copy) }

  /**
   * The Forward.
   *
   * @param input the input sequence to encode
   *
   * @return a list containing the forwarded sequence for each network
   */
  override fun forward(input: List<InputNDArrayType>): List<List<DenseNDArray>> {

    val encodersOutputs: List<List<DenseNDArray>> = this.encoders.map { it.forward(input) }

    return List(
      size = input.size,
      init = { elementIndex ->
        List(size = this.encoders.size, init = { encoderIndex -> encodersOutputs[encoderIndex][elementIndex] })
      }
    )
  }

  /**
   * Execute the backward for each element of the input sequence, given sequence of output errors (one for
   * each network), and return its input errors.
   *
   * @param outputErrors the sequence of output errors to propagate
   */
  override fun backward(outputErrors: List<List<DenseNDArray>>) {

    this.encoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(outputErrors.map { it[encoderIndex] } )
    }
  }
}
