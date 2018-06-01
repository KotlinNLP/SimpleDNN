/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.encoders.sequenceencoder

import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A Feedforward Encoder with multiple parallel outputs for each input.
 * It encodes a sequence of arrays into another sequence of n parallel arrays using more networks.
 *
 * @param model the model of the sequence parallel encoder
 */
class SequenceParallelEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(val model: ParallelEncoderModel) {

  /**
   * A list of processors which encode each input array into multiple vectors.
   */
  private val encoders: List<BatchFeedforwardProcessor<InputNDArrayType>> = this.model.networks.map { network ->
    BatchFeedforwardProcessor<InputNDArrayType>(network)
  }

  /**
   * @return the errors of the input sequence
   */
  fun getInputSequenceErrors(): Array<DenseNDArray> {

    val inputErrors: Array<DenseNDArray> = this.encoders[0].getBatchInputErrors(copy = true)

    for (encoderIndex in 1 until (this.model.networks.size - 1)) {
      inputErrors.zip(this.encoders[encoderIndex].getBatchInputErrors(copy = false)).forEach {
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
  fun getParamsErrors(copy: Boolean = true)
    = ParallelEncoderParameters(this.encoders.map { it.getParamsErrors(copy = copy) })

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   *
   * @return a list containing the forwarded sequence for each network
   */
  fun encode(sequence: Array<InputNDArrayType>): List<List<DenseNDArray>> = this.forward(sequence)

  /**
   * Execute the backward for each element of the input sequence, given sequence of output errors (one for
   * each network), and return its input errors.
   *
   * @param outputErrorsSequence the sequence of output errors to propagate, grouped for each network
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequence: Array<Array<DenseNDArray>>, propagateToInput: Boolean) {

    this.encoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(
        Array(
          size = outputErrorsSequence.size,
          init = { elementIndex -> outputErrorsSequence[elementIndex][encoderIndex] }
        ),
        propagateToInput = propagateToInput)
    }
  }

  /**
   * Forward each array of the [sequence] within each feed-forward network.
   *
   * @param sequence the sequence to forward
   *
   * @return a list containing the forwarded sequence for each network
   */
  private fun forward(sequence: Array<InputNDArrayType>): List<List<DenseNDArray>> {

    val encodersOutputs: List<Array<DenseNDArray>> = this.encoders.map { it.forward(sequence) }

    return List(
      size = sequence.size,
      init = { elementIndex ->
        List(size = this.encoders.size, init = { encoderIndex -> encodersOutputs[encoderIndex][elementIndex] })
      }
    )
  }
}
