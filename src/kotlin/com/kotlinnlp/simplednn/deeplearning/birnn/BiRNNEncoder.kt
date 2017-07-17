/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * Bidirectional Recursive Neural Network Encoder
 *
 * The BiRNNEncoder encodes a sequence (i.e., words) using a concatenation of two RNNs,
 * one processing from left-to-right and the other processing right-to-left.
 * The result is a vector representation for each element which captures information
 * of the element itself and an “infinite” window around it.
 *
 * This implementation support a sequence encoding at time.
 *
 * @property network the [BiRNN] of this encoder
 */
class BiRNNEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(private val network: BiRNN) {

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence left-to-right.
   */
  private val leftToRightProcessor = RecurrentNeuralProcessor<InputNDArrayType>(this.network.leftToRightNetwork)

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence right-to-left.
   */
  private val rightToLeftProcessor = RecurrentNeuralProcessor<InputNDArrayType>(this.network.rightToLeftNetwork)

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<InputNDArrayType>): Array<DenseNDArray> {

    val (leftToRightOut, rightToLeftOut) = this.biEncoding(sequence)

    return BiRNNUtils.concatenate(leftToRightOut, rightToLeftOut)
  }

  /**
   * Propagate the errors of the entire sequence.
   *
   * @param outputErrorsSequence the errors to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean) {

    val (leftToRightOutputErrors, rightToLeftOutputErrors) =
      BiRNNUtils.splitErrorsSequence(outputErrorsSequence)

    this.leftToRightProcessor.backward(
      outputErrorsSequence = leftToRightOutputErrors,
      propagateToInput = propagateToInput)

    this.rightToLeftProcessor.backward(
      outputErrorsSequence = rightToLeftOutputErrors.reversed().toTypedArray(),
      propagateToInput = propagateToInput)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence (the errors of the two RNNs are combined by summation)
   */
  fun getInputSequenceErrors(copy: Boolean = true): Array<DenseNDArray> {
    return BiRNNUtils.sumBidirectionalErrors(
      leftToRightInputErrors = this.leftToRightProcessor.getInputSequenceErrors(copy = copy),
      rightToLeftInputErrors = this.rightToLeftProcessor.getInputSequenceErrors(copy = copy)
    )
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the BiRNN parameters
   */
  fun getParamsErrors(copy: Boolean = true): BiRNNParameters {
    return BiRNNParameters(
      leftToRight = leftToRightProcessor.getParamsErrors(copy = copy),
      rightToLeft = rightToLeftProcessor.getParamsErrors(copy = copy)
    )
  }

  /**
   * Given a [sequence] return the encoded left-to-right and right-to-left representation.
   *
   * @param sequence the sequence to encode
   *
   * @return a Pair with two arrays containing the outputs of the two RNNs
   */
  private fun biEncoding(sequence: Array<InputNDArrayType>):
    Pair<Array<DenseNDArray>, Array<DenseNDArray>> {

    val leftToRightOut = arrayOfNulls<DenseNDArray>(sequence.size)
    val rightToLeftOut = arrayOfNulls<DenseNDArray>(sequence.size)

    var isFirstElement: Boolean = true

    sequence.indices.zip(sequence.indices.reversed()).forEach { (i, r) ->
      leftToRightOut[i] = this.leftToRightProcessor.forward(sequence[i], firstState = isFirstElement)
      rightToLeftOut[r] = this.rightToLeftProcessor.forward(sequence[r], firstState = isFirstElement)

      isFirstElement = false
    }

    return Pair(
      leftToRightOut.requireNoNulls(),
      rightToLeftOut.requireNoNulls()
    )
  }
}
