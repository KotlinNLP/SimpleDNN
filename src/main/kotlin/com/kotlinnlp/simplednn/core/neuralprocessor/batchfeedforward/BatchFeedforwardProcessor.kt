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
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
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
  private val processorsPool = FeedforwardNeuralProcessorsPool<InputNDArrayType>(neuralNetwork)

  /**
   * Contains the errors accumulated from the processors during the forward.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

  /**
   * The amount of processors used at a given time.
   */
  private var usedProcessors: MutableList<FeedforwardNeuralProcessor<InputNDArrayType>> = mutableListOf()

  /**
   * Get the input errors of all the batch.
   * This method must be used when the input layer is not a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  fun getInputErrors(copy: Boolean = true): List<DenseNDArray> =
    this.usedProcessors.map { it.getInputErrors(copy = copy) }

  /**
   * Get the inputs errors of all the batch.
   * This method must be used when the input layer is a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the list of errors of the inputs
   */
  fun getInputsErrors(copy: Boolean = true): List<List<DenseNDArray>> =
    this.usedProcessors.map { it.getInputsErrors(copy = copy) }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the internal network
   */
  override fun getParamsErrors(copy: Boolean): NetworkParameters
    = this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Forward each array of the [featuresBatch] within a dedicated feed-forward processor.
   * This method must be used when the input layer is not a Merge layer.
   *
   * If [continueBatch] is true, the current forwarding batch is expanded with the new elements given,
   * otherwise a new batch is started (the default).
   *
   * @param featuresBatch the batch to forward
   * @param continueBatch whether this batch is the continuation of the last forwarded one (without have called a
   *        backward)
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(featuresBatch: List<InputNDArrayType>, continueBatch: Boolean = false): List<DenseNDArray> =
    featuresBatch.mapIndexed { i, features ->
      if (!continueBatch && i == 0) this.reset()
      this.forwardProcessor(features)
    }

  /**
   * Forward the inputs with a dedicated feed-forward processor.
   * This method must be used when the input layer is a Merge layer.
   *
   * If [continueBatch] is true, the current forwarding batch is expanded with the new elements given,
   * otherwise a new batch is started (the default).
   *
   * @param featuresListBatch the batch to forward
   * @param continueBatch whether this batch is the continuation of the last forwarded one (without have called a
   *        backward)
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(featuresListBatch: ArrayList<List<InputNDArrayType>>,
              continueBatch: Boolean = false): List<DenseNDArray> =
    featuresListBatch.mapIndexed { i, featuresList ->
      if (!continueBatch && i == 0) this.reset()
      this.forwardProcessor(featuresList)
    }

  /**
   * Execute the backward for a single element of the input batch, given its output errors and the index within the
   * range of all the elements of the batch.
   *
   * @param elementIndex the index of an element of the whole batch
   * @param outputErrors the output errors of a single element of the batch
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(elementIndex: Int, outputErrors: DenseNDArray, propagateToInput: Boolean) {

    require(elementIndex in 0 until this.usedProcessors.size) {
      "The processor index exceeds the last index of the used processors."
    }

    this.processorBackward(
      processor = this.usedProcessors[elementIndex],
      errors = outputErrors,
      propagateToInput = propagateToInput)
  }

  /**
   * Execute the backward for each element of the input batch, given its output errors.
   *
   * @param outputErrors the output errors of the batch to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrors: List<DenseNDArray>, propagateToInput: Boolean) {

    require(outputErrors.size == this.usedProcessors.size) {
      "Number of errors (%d) does not reflect the number of used processors (%d)".format(
        outputErrors.size, this.usedProcessors.size)
    }

    this.usedProcessors.zip(outputErrors).forEach { (processor, errors) ->
      this.processorBackward(processor = processor, errors = errors, propagateToInput = propagateToInput)
    }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Forward the input with a dedicated feed-forward processor, when the input layer is not a Merge layer.
   *
   * @param features the input features
   *
   * @return an array containing the forwarded sequence
   */
  private fun forwardProcessor(features: InputNDArrayType): DenseNDArray =
    this.processorsPool.getItem().let {
      this.usedProcessors.add(it)
      it.forward(features)
    }

  /**
   * Forward the input with a dedicated feed-forward processor, when the input layer is a Merge layer.
   *
   * @param featuresList the list of input features
   *
   * @return an array containing the forwarded sequence
   */
  private fun forwardProcessor(featuresList: List<InputNDArrayType>): DenseNDArray =
    this.processorsPool.getItem().let {
      this.usedProcessors.add(it)
      it.forward(featuresList)
    }

  /**
   * Reset temporary memories.
   */
  private fun reset() {
    this.processorsPool.releaseAll()
    this.usedProcessors.clear()
    this.errorsAccumulator.reset()
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
