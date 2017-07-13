/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
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
   * The [FeedforwardNeuralProcessor] which merges the outputs of the input RNNs into a single vector.
   */
  private val outputProcessorList = ArrayList<FeedforwardNeuralProcessor<DenseNDArray>>()

  /**
   * Contains the errors accumulated from the outputProcessorList during the encoding process.
   */
  private val outputErrorsAccumulator = ParamsErrorsAccumulator(this.network.outputNetwork)

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<InputNDArrayType>): Array<DenseNDArray> {

    this.reset()

    val (leftToRightOut, rightToLeftOut) = this.biEncoding(sequence)

    return this.forwardOutput(BiRNNUtils.concatenate(leftToRightOut, rightToLeftOut))
  }

  /**
   * Propagate the errors of the entire sequence.
   * Accumulate the errors of the parameters into the optimizer.
   *
   * @param outputErrorsSequence the errors to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   *
   * @return the errors of the parameters of the [network]
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean): BiRNNParameters {

    require(outputErrorsSequence.size == outputProcessorList.size) {
      "Number of errors (%d) does not reflect the number of processed item (%s)".format(
        outputErrorsSequence.size, outputProcessorList.size)
    }

    this.RNNsBackward(
      outputErrorsSequence = this.outputBackward(outputErrorsSequence),
      propagateToInput = propagateToInput)

    return this.getParamsErrors()
  }

  /**
   * @return the errors of the input sequence (the errors of the two RNNs are combined by summation)
   */
  fun getInputSequenceErrors(): Array<DenseNDArray> {
    return BiRNNUtils.sumBidirectionalErrors(
      leftToRightInputErrors = this.leftToRightProcessor.getInputSequenceErrors(),
      rightToLeftInputErrors = this.rightToLeftProcessor.getInputSequenceErrors()
    )
  }

  /**
   * Add a new output processor
   *
   * @return the new added output processor
   */
  private fun addNewOutputProcessor(): FeedforwardNeuralProcessor<DenseNDArray> {
    this.outputProcessorList.add(FeedforwardNeuralProcessor(this.network.outputNetwork))
    return this.outputProcessorList.last()
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

  /**
   * Forward the results of the input RNNs within the feed-forward output processor.
   *
   * @param sequence the sequence to forward
   *
   * @return an array containing the forwarded sequence
   */
  private fun forwardOutput(sequence: Array<DenseNDArray>): Array<DenseNDArray> =
    Array(size = sequence.size, init = {
      this.addNewOutputProcessor().forward(sequence[it])
    })

  /**
   * Execute the backward of the output processor and return its input errors.
   *
   * @param outputErrorsSequence the errors to propagate
   *
   * @return the errors to propagate to the two RNNs
   */
  private fun outputBackward(outputErrorsSequence: Array<DenseNDArray>): Array<DenseNDArray> {

    require(outputErrorsSequence.size == outputProcessorList.size) {
      "Number of errors (%d) does not reflect the length of the number of mlp processors (%d)".format(
        outputErrorsSequence.size, outputProcessorList.size)
    }

    val inputErrors = Array(size = outputErrorsSequence.size, init = {

      this.outputProcessorList[it].backward(outputErrors = outputErrorsSequence[it], propagateToInput = true)

      this.outputErrorsAccumulator.accumulate(this.outputProcessorList[it].getParamsErrors())

      this.outputProcessorList[it].getInputErrors()
    })

    this.outputErrorsAccumulator.averageErrors()

    return inputErrors
  }

  /**
   * Propagate the errors to the input RNNs.
   *
   * @param outputErrorsSequence the sequence errors to propagate
   * @param propagateToInput whether to propagate the errors to the input
   */
  private fun RNNsBackward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean) {

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
   * Return the errors of the BiRNNParameters
   *
   * @return the errors of the BiRNN parameters
   */
  private fun getParamsErrors(): BiRNNParameters {
    return BiRNNParameters(
      leftToRight = leftToRightProcessor.getParamsErrors(),
      rightToLeft = rightToLeftProcessor.getParamsErrors(),
      output = outputErrorsAccumulator.getParamsErrors()
    )
  }

  /**
   * Reset temporary memories
   */
  private fun reset() {
    this.outputProcessorList.clear()
    this.outputErrorsAccumulator.reset()
  }
}
