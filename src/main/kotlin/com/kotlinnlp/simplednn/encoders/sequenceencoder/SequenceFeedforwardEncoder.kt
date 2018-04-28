/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.encoders.sequenceencoder

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Sequence Feedforward Encoder.
 *
 * It encodes a sequence of arrays into another sequence of arrays using a [SequenceFeedforwardNetwork].
 */
class SequenceFeedforwardEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val network: SequenceFeedforwardNetwork
) {

  /**
   * A list of [FeedforwardNeuralProcessor]s which merge each input array into a single vector.
   */
  private val processorsList = ArrayList<FeedforwardNeuralProcessor<InputNDArrayType>>()

  /**
   * Contains the errors accumulated from the [processorsList] during the encoding process.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

  /**
   * The amount of processors used at a given time.
   */
  private var usedProcessors: Int = 0

  /**
   * The index of the current processor.
   */
  private var curProcessorIndex: Int = -1

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence
   */
  fun getInputSequenceErrors(copy: Boolean = true): Array<DenseNDArray> = Array(
    size = this.usedProcessors,
    init = { i -> this.processorsList[i].getInputErrors(copy = copy) }
  )

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the internal network
   */
  fun getParamsErrors(copy: Boolean = true): NetworkParameters
    = this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<InputNDArrayType>): Array<DenseNDArray> {

    this.reset()

    return this.forward(sequence)
  }

  /**
   * Forward the input with a dedicated feed-forward processor.
   *
   * @param input the input
   * @param firstState true if the [input] is the first of a sequence
   *
   * @return an array containing the forwarded sequence
   */
  fun encode(input: InputNDArrayType, firstState: Boolean): DenseNDArray {

    if (firstState) this.reset()

    val processor = this.getProcessor(++this.curProcessorIndex)

    this.usedProcessors++

    return processor.forward(input)
  }

  /**
   * Execute the backward on the processor at the [curProcessorIndex].
   *
   * @param outputErrors the output errors of the last used processor
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backwardStep(outputErrors: DenseNDArray, propagateToInput: Boolean) {

    require(this.curProcessorIndex >= 0)

    this.processorBackward(
      processor = this.processorsList[this.curProcessorIndex],
      errors = outputErrors,
      propagateToInput = propagateToInput)

    if (this.curProcessorIndex == 0) {
      this.errorsAccumulator.averageErrors()
    }

    this.curProcessorIndex--
  }

  /**
   * Execute the backward for each element of the input sequence, given its output errors, and return its input errors.
   *
   * @param outputErrorsSequence the output errors of the sequence to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean) {

    require(outputErrorsSequence.size == this.usedProcessors) {
      "Number of errors (%d) does not reflect the number of used processors (%d)".format(
        outputErrorsSequence.size, this.usedProcessors)
    }

    for (i in 0 until this.usedProcessors) {
      this.processorBackward(
        processor = this.processorsList[i],
        errors = outputErrorsSequence[i],
        propagateToInput = propagateToInput)

      this.curProcessorIndex--
    }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Reset temporary memories.
   */
  private fun reset() {
    this.usedProcessors = 0
    this.curProcessorIndex = -1
    this.errorsAccumulator.reset()
  }

  /**
   * Forward each array of the [sequence] within a dedicated feed-forward processor.
   *
   * @param sequence the sequence to forward
   *
   * @return an array containing the forwarded sequence
   */
  private fun forward(sequence: Array<InputNDArrayType>): Array<DenseNDArray> =
    Array(
      size = sequence.size,
      init = { i ->
        val processor = this.getProcessor(i)

        this.usedProcessors++
        this.curProcessorIndex++

        processor.forward(sequence[i])
      }
    )

  /**
   * Get the processor at the given index (create a new one if the index exceeds the size of the list).
   *
   * @param index the index of the processor in the list
   *
   * @return the processor at the given [index]
   */
  private fun getProcessor(index: Int): FeedforwardNeuralProcessor<InputNDArrayType> {

    require(index <= this.processorsList.size) {
      "Invalid output processor index: %d (size = %d)".format(index, this.processorsList.size)
    }

    if (index == this.processorsList.size) { // add a new processor into the list
      this.processorsList.add(FeedforwardNeuralProcessor(this.network.network))
    }

    return this.processorsList[index]
  }

  /**
   * Perform the backward of a processor by the given [errors] and accumulate its params errors.
   *
   * @param processor the processor to perform the backward
   * @param errors the output errors of the [processor]
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  private fun processorBackward(processor: FeedforwardNeuralProcessor<InputNDArrayType>,
                                errors: DenseNDArray,
                                propagateToInput: Boolean) {

    processor.backward(outputErrors = errors, propagateToInput = propagateToInput)

    this.errorsAccumulator.accumulate(processor.getParamsErrors(copy = false))
  }
}
