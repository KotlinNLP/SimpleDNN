/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A Sequence Feedforward Encoder with multiple parallel outputs for each input.
 * It encodes a sequence of arrays into another sequence of n parallel arrays using more [SequenceFeedforwardNetwork]s.
 *
 * @param model the model of the sequence parallel encoder
 */
class SequenceParallelEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(val model: ParallelEncoderModel) {

  /**
   * A list of [SequenceFeedforwardEncoder]s which encode each input array into multiple vectors.
   */
  private val encoders: List<SequenceFeedforwardEncoder<InputNDArrayType>> = this.model.networks.map { network ->
    SequenceFeedforwardEncoder<InputNDArrayType>(network)
  }

  /**
   * @return the errors of the input sequence
   */
  fun getInputSequenceErrors(): Array<DenseNDArray> {

    val inputErrors: Array<DenseNDArray> = this.encoders[0].getInputSequenceErrors(copy = true)

    for (encoderIndex in 1 until (this.model.networks.size - 1)) {
      inputErrors.zip(this.encoders[encoderIndex].getInputSequenceErrors(copy = false)).forEach {
        (baseErrors, errors) -> baseErrors.assignSum(errors)
      }
    }

    return inputErrors
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the internal network
   */
  fun getParamsErrors(copy: Boolean = true): Array<NetworkParameters> = Array(
    size = this.model.networks.size,
    init = { i -> this.encoders[i].getParamsErrors(copy = copy) }
  )

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<InputNDArrayType>): Array<Array<DenseNDArray>> {
    return this.forward(sequence)
  }

  /**
   * Execute the backward for each element of the input sequence, given the array of output errors sequences (one for
   * each network), and return its input errors.
   *
   * @param outputErrorsSequences the output errors of the sequence to propagate, grouped for each network
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequences: Array<Array<DenseNDArray>>, propagateToInput: Boolean) {

    this.encoders.forEachIndexed { i, it ->
      it.backward(
        outputErrorsSequence = Array(size = outputErrorsSequences.size, init = { k -> outputErrorsSequences[k][i] }),
        propagateToInput = propagateToInput)
    }
  }

  /**
   * Forward each array of the [sequence] within each feed-forward network.
   *
   * @param sequence the sequence to forward
   *
   * @return an array containing one forwarded sequence for each network
   */
  private fun forward(sequence: Array<InputNDArrayType>): Array<Array<DenseNDArray>> {

    val outputsPerEncoder: Array<Array<DenseNDArray>> = Array(
      size = this.model.networks.size,
      init = { i -> this.encoders[i].encode(sequence) }
    )

    return Array(
      size = sequence.size,
      init = { elementIndex ->
        Array(size = this.encoders.size, init = { encoderIndex -> outputsPerEncoder[encoderIndex][elementIndex] })
      }
    )
  }
}
