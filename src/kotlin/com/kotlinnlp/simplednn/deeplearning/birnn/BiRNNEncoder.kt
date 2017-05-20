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
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNUtils.sumBidirectionalErrors
import com.kotlinnlp.simplednn.simplemath.NDArray

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
 * @property optimizer the Optimizer associated to this BiRNN (can be null)
 */
class BiRNNEncoder(
  private val network: BiRNN,
  private val optimizer: BiRNNOptimizer?) {

  /**
   * The [RecurrentNeuralProcessor] to encode the sequence left-to-right
   */
  private val leftToRightProcessor = RecurrentNeuralProcessor(this.network.leftToRightNetwork)

  /**
   * The [RecurrentNeuralProcessor] to encode the sequence right-to-left
   */
  private val rightToLeftProcessor = RecurrentNeuralProcessor(this.network.rightToLeftNetwork)

  /**
   * The [FeedforwardNeuralProcessor] to take two RNN vectors and returning a single d-dimensional vector
   */
  private val outputProcessorList = ArrayList<FeedforwardNeuralProcessor>()

  /**
   * Contains the errors accumulated from the outputProcessorList during the encoding process
   */
  private val outputErrorsAccumulator = ParamsErrorsAccumulator(this.network.outputNetwork)

  /**
   * Encode the [sequence]
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<NDArray>): Array<NDArray> {

    this.reset()

    val (leftToRightOut, rightToLeftOut) = biEncoding(sequence)

    return this.feedForward(BiRNNUtils.concatenate(leftToRightOut, rightToLeftOut))
  }

  /**
   * Encode the [sequence]
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: ArrayList<NDArray>): Array<NDArray> = this.encode(sequence.toTypedArray())

  /**
   * Propagate the errors of the entire sequence.
   * Accumulate the the params errors on the optimizer (required).
   *
   * @param outputErrorsSequence the errors to propagate
   * @params propagateToInput whether to propagate the output errors to the input or not
   */
  fun propagateErrors(outputErrorsSequence: Array<NDArray>, propagateToInput: Boolean) {

    require(this.optimizer != null){
      "Impossible to propagate errors without an optimizer"
    }

    require(outputErrorsSequence.size == outputProcessorList.size) {
      "Number of errors (${outputErrorsSequence.size}) does not reflect the number of processed item (${outputProcessorList.size})" }

    this.propagateErrorsOnRNNs(
      outputErrorsSequence = this.propagateErrorsOnOutput(outputErrorsSequence),
      propagateToInput = propagateToInput)

    this.optimizer!!.accumulate(this.getParamsErrors())
  }

  /**
   * Return the errors of the sequence.
   *
   * The errors of the two recurrent networks are combined by summation.
   *
   * @return the errors of the sequence
   */
  fun getInputSequenceErrors(): Array<NDArray> {
    return sumBidirectionalErrors(
      leftToRightInputErrors = this.leftToRightProcessor.getInputSequenceErrors(),
      rightToLeftInputErrors = this.rightToLeftProcessor.getInputSequenceErrors()
    )
  }

  /**
   * Add a new output processor
   *
   * @return the new added output processor
   */
  private fun addNewOutputProcessor(): FeedforwardNeuralProcessor {
    this.outputProcessorList.add(FeedforwardNeuralProcessor(this.network.outputNetwork))
    return this.outputProcessorList.last()
  }

  /**
   * Given a [sequence] return the encoded left-to-right and right-to-left representation
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence of the form Pair<leftToRightOut, rightToLeftOut>
   */
  private fun biEncoding(sequence: Array<NDArray>): Pair<Array<NDArray>, Array<NDArray>> {

    val leftToRightOut = arrayOfNulls<NDArray>(sequence.size)
    val rightToLeftOut = arrayOfNulls<NDArray>(sequence.size)

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
   * Return the results of the feed-forward output processor on the entire [sequence]
   *
   * @param sequence the sequence to forward
   *
   * @return the results of the forward
   */
  private fun feedForward(sequence: Array<NDArray>): Array<NDArray> =
    Array(size = sequence.size, init = {
      this.addNewOutputProcessor().forward(sequence[it])
    })

  /**
   * Propagate the errors on the output processor and return its input errors.
   *
   * @param outputErrorsSequence the errors to propagate
   *
   * @return the errors to propagate on the two recurrent networks
   */
  private fun propagateErrorsOnOutput(outputErrorsSequence: Array<NDArray>): Array<NDArray> {

    require(outputErrorsSequence.size == outputProcessorList.size) {
      "Number of errors (${outputErrorsSequence.size}) does not reflect the length of the number of mlp processors (${outputProcessorList.size})"
    }

    return Array(size = outputErrorsSequence.size, init = {

      this.outputProcessorList[it].backward(outputErrors = outputErrorsSequence[it], propagateToInput = true)

      this.outputErrorsAccumulator.accumulate(this.outputProcessorList[it].getParamsErrors())

      this.outputProcessorList[it].getInputErrors()
    })
  }

  /**
   * Propagate the errors on the two recurrent networks
   *
   * @param outputErrorsSequence the sequence errors to propagate
   * @param propagateToInput whether to propagate the errors on the input
   */
  private fun propagateErrorsOnRNNs(outputErrorsSequence: Array<NDArray>, propagateToInput: Boolean) {

    val (leftToRightOutputErrors, rightToLeftOutputErrors) =
      BiRNNUtils.splitErrorsSequence(outputErrorsSequence)

    this.leftToRightProcessor.backward(
      outputErrorsSequence = leftToRightOutputErrors,
      propagateToInput = propagateToInput)

    this.rightToLeftProcessor.backward(
      outputErrorsSequence = rightToLeftOutputErrors,
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
