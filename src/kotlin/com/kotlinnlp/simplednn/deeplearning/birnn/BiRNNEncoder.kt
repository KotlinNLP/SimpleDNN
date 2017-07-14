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
   * A list of [FeedforwardNeuralProcessor]s which merge each pair of output arrays of the input RNNs into a single
   * vector.
   */
  private val outputProcessorsList = ArrayList<FeedforwardNeuralProcessor<DenseNDArray>>()

  /**
   * Contains the errors accumulated from the outputProcessorsList during the encoding process.
   */
  private val outputErrorsAccumulator = ParamsErrorsAccumulator(this.network.outputNetwork)

  /**
   * The amount of processors used at a given time.
   */
  private var outputProcessorsListSize: Int = 0

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
   *
   * @param outputErrorsSequence the errors to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean) {

    require(outputErrorsSequence.size == this.outputProcessorsListSize) {
      "Number of errors (%d) does not reflect the number of processed item (%s)".format(
        outputErrorsSequence.size, this.outputProcessorsListSize)
    }

    this.RNNsBackward(
      outputErrorsSequence = this.outputBackward(outputErrorsSequence),
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
      rightToLeft = rightToLeftProcessor.getParamsErrors(copy = copy),
      output = outputErrorsAccumulator.getParamsErrors(copy = copy)
    )
  }

  /**
   * Get the output processor at the given index (create a new one if the index exceeds the size of the list).
   *
   * @param index the index of the output processor in the lis
   *
   * @return the output processor at the given [index]
   */
  private fun getOutputProcessor(index: Int): FeedforwardNeuralProcessor<DenseNDArray> {

    require(index <= this.outputProcessorsListSize) {
      "Invalid output processor index: %d (size = %d)".format(index, this.outputProcessorsListSize)
    }

    if (index == this.outputProcessorsListSize) { // add a new processor into the list
      this.outputProcessorsList.add(FeedforwardNeuralProcessor(this.network.outputNetwork))
      this.outputProcessorsListSize++
    }

    return this.outputProcessorsList[index]
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
    Array(
      size = sequence.size,
      init = { i -> this.getOutputProcessor(i).forward(sequence[i]) }
    )

  /**
   * Execute the backward of the output processor and return its input errors.
   *
   * @param outputErrorsSequence the errors to propagate
   *
   * @return the errors to propagate to the two RNNs
   */
  private fun outputBackward(outputErrorsSequence: Array<DenseNDArray>): Array<DenseNDArray> {

    require(outputErrorsSequence.size == this.outputProcessorsListSize) {
      "Number of errors (%d) does not reflect the length of the number of mlp processors (%d)".format(
        outputErrorsSequence.size, this.outputProcessorsListSize)
    }

    val inputErrors = Array(
      size = outputErrorsSequence.size,
      init = { i ->
        this.outputProcessorBackward(
          processor = this.outputProcessorsList[i],
          errors = outputErrorsSequence[i])
      })

    this.outputErrorsAccumulator.averageErrors()

    return inputErrors
  }

  /**
   * Perform the backward of an output processor from the given [errors] and accumulate its params errors.
   *
   * @param processor the output processor
   * @param errors the output errors of the [processor]
   *
   * @return the input errors by reference
   */
  private fun outputProcessorBackward(processor: FeedforwardNeuralProcessor<DenseNDArray>,
                                      errors: DenseNDArray): DenseNDArray {

    processor.backward(outputErrors = errors, propagateToInput = true)

    this.outputErrorsAccumulator.accumulate(processor.getParamsErrors())

    return processor.getInputErrors(copy = false)
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
   * Reset temporary memories.
   */
  private fun reset() {
    this.outputProcessorsListSize = 0
    this.outputErrorsAccumulator.reset()
  }
}
