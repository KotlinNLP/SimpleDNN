/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [BatchFeedforwardProcessor] acts on the [neuralNetwork] performing predictions and training based on batch of
 * examples.
 *
 * @property neuralNetwork a [NeuralNetwork]
 * @property id an identification number useful to track a specific processor
 */
class BatchFeedforwardProcessor<InputNDArrayType: NDArray<InputNDArrayType>>(
  neuralNetwork: NeuralNetwork,
  id: Int = 0
) : NeuralProcessor(neuralNetwork = neuralNetwork, id = id) {

  /**
   * A list of processors, one for each element of the batch.
   */
  private val processorsList = mutableListOf<FeedforwardNeuralProcessor<InputNDArrayType>>()

  /**
   * Contains the errors accumulated from the [processorsList] during the forward process.
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
   * Get the input errors of all the batch.
   * This method must be used when the input layer is not a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  fun getBatchInputErrors(copy: Boolean = true): List<DenseNDArray> = List(
    size = this.usedProcessors,
    init = { i -> this.processorsList[i].getInputErrors(copy = copy) }
  )

  /**
   * Get the inputs errors of all the batch.
   * This method must be used when the input layer is a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the list of errors of the inputs
   */
  fun getInputsErrors(copy: Boolean = true): List<List<DenseNDArray>> = List(
    size = this.usedProcessors,
    init = { i -> this.processorsList[i].getInputsErrors(copy = copy) }
  )

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the internal network
   */
  override fun getParamsErrors(copy: Boolean): NetworkParameters
    = this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Forward each array of the [featuresBatch] within a dedicated feed-forward processor.
   *
   * @param featuresBatch the batch to forward
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(featuresBatch: List<InputNDArrayType>): List<DenseNDArray> =
    featuresBatch.mapIndexed { i, features -> this.forward(features, firstState = i == 0) }

  /**
   * Forward the inputs with a dedicated feed-forward processor.
   * This method must be used when the input layer is a Merge layer.
   *
   * @param featuresListBatch the batch to forward
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(featuresListBatch: ArrayList<List<InputNDArrayType>>): List<DenseNDArray> =
    featuresListBatch.mapIndexed { i, featuresList -> this.forward(featuresList, firstState = i == 0) }

  /**
   * Execute the backward for each element of the input batch, given its output errors.
   *
   * @param outputErrors the output errors of the batch to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrors: List<DenseNDArray>, propagateToInput: Boolean) {

    require(outputErrors.size == this.usedProcessors) {
      "Number of errors (%d) does not reflect the number of used processors (%d)".format(
        outputErrors.size, this.usedProcessors)
    }

    for (i in 0 until this.usedProcessors) {
      this.processorBackward(
        processor = this.processorsList[i],
        errors = outputErrors[i],
        propagateToInput = propagateToInput)

      this.curProcessorIndex--
    }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Forward the input with a dedicated feed-forward processor, when the input layer is not a Merge layer.
   *
   * @param features the input features
   * @param firstState true if the [features] is the first of a sequence
   *
   * @return an array containing the forwarded sequence
   */
  private fun forward(features: InputNDArrayType, firstState: Boolean): DenseNDArray {

    if (firstState) this.reset()

    this.curProcessorIndex++
    this.usedProcessors++

    return this.getProcessor(this.curProcessorIndex).forward(features)
  }

  /**
   * Forward the input with a dedicated feed-forward processor, when the input layer is a Merge layer.
   *
   * @param featuresList the list of input features
   * @param firstState true if the [featuresList] is the first of a sequence
   *
   * @return an array containing the forwarded sequence
   */
  private fun forward(featuresList: List<InputNDArrayType>, firstState: Boolean): DenseNDArray {

    if (firstState) this.reset()

    this.curProcessorIndex++
    this.usedProcessors++

    return this.getProcessor(this.curProcessorIndex).forward(featuresList)
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
      this.processorsList.add(FeedforwardNeuralProcessor(this.neuralNetwork))
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
