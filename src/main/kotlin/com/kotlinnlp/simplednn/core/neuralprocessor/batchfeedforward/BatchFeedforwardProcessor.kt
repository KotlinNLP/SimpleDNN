/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The neural processor that acts on networks of stacked-layers, performing operations through with mini-batches.
 *
 * @property model the stacked-layers parameters
 * @property useDropout whether to apply the dropout during the [forward]
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property paramsErrorsCollector where to collect the local params errors during the [backward] (optional)
 * @property id an identification number useful to track a specific processor
 */
class BatchFeedforwardProcessor<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: StackedLayersParameters,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * The errors of the parameters which will be filled at each [backward].
   */
  private val paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector()

  /**
   * Contains the errors accumulated from the processors during the forward.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator()

  /**
   * A list of processors, one for each element of the batch.
   */
  private val processorsPool = FeedforwardNeuralProcessorsPool<InputNDArrayType>(
    model = this.model,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput,
    paramsErrorsCollector = this.paramsErrorsCollector)

  /**
   * The processors currently used (as many as the current batch size).
   */
  private var usedProcessors: MutableList<FeedforwardNeuralProcessor<InputNDArrayType>> = mutableListOf()

  /**
   * Get the input errors of the whole batch when the input layer is NOT a merge layer.
   *
   * @param copy whether the returned errors must be a copy or a reference
   *
   * @return the input errors of the whole batch
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    this.usedProcessors.map { it.getInputErrors(copy = copy) }

  /**
   * Get the inputs errors of the whole batch when the input layer is a merge layer.
   *
   * @param copy whether the returned errors must be a copy or a reference
   *
   * @return the input errors of the whole batch
   */
  fun getInputsErrors(copy: Boolean = true): List<List<DenseNDArray>> =
    this.usedProcessors.map { it.getInputsErrors(copy = copy) }

  /**
   * @param copy whether the returned errors must be a copy or a reference
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Execute the forward of the input to the output, using a dedicated feed-forward processor for each array of the
   * batch.
   * This method must be used when the input layer is NOT a merge layer.
   *
   * @param input the input batch
   *
   * @return a list containing the output of each forwarded processor
   */
  override fun forward(input: List<InputNDArrayType>): List<DenseNDArray> =
    this.forward(input = input, continueBatch = false)

  /**
   * Execute the forward of the input to the output, using a dedicated feed-forward processor for each array of the
   * batch.
   * This method must be used when the input layer is NOT a merge layer.
   *
   * If [continueBatch] is `true`, the current forwarding batch is expanded with the given [input], otherwise a new
   * batch is started (the default).
   *
   * WARNING: [continueBatch] should not be `true` if a backward has been called before.
   *
   * @param input the input batch
   * @param continueBatch whether this batch is the continuation of the last forwarded one
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(input: List<InputNDArrayType>, continueBatch: Boolean = false): List<DenseNDArray> =
    input.mapIndexed { i, values ->
      if (!continueBatch && i == 0) this.reset()
      this.forwardProcessor(values)
    }

  /**
   * Execute the forward of the input to the output, using a dedicated feed-forward processor for each array of the
   * batch.
   * This method must be used when the input layer is a merge layer.
   *
   * If [continueBatch] is `true`, the current forwarding batch is expanded with the given [input], otherwise a new
   * batch is started (the default).
   *
   * WARNING: [continueBatch] should not be `true` if a backward has been called before.
   *
   * @param input the input batch
   * @param continueBatch whether this batch is the continuation of the last forwarded one
   *
   * @return a list containing the output of each forwarded processor
   */
  fun forward(input: ArrayList<List<InputNDArrayType>>, continueBatch: Boolean = false): List<DenseNDArray> =
    input.mapIndexed { i, featuresList ->
      if (!continueBatch && i == 0) this.reset()
      this.forwardProcessor(featuresList)
    }

  /**
   * Execute the backward for a single element of the input batch, given its output errors and its index within the
   * range of all the elements of the current batch.
   *
   * @param elementIndex the index of an element within the whole batch
   * @param outputErrors the output errors given element
   */
  fun backward(elementIndex: Int, outputErrors: DenseNDArray) {

    require(elementIndex in 0 until this.usedProcessors.size) {
      "The processor index exceeds the last index of the used processors."
    }

    this.processorBackward(processor = this.usedProcessors[elementIndex], outputErrors = outputErrors)
  }

  /**
   * Execute the backward for all the elements of the batch, given the output errors.
   *
   * @param outputErrors the output errors of the whole batch
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    require(outputErrors.size == this.usedProcessors.size) {
      "Number of errors (%d) does not reflect the number of used processors (%d)".format(
        outputErrors.size, this.usedProcessors.size)
    }

    this.usedProcessors.zip(outputErrors).forEach { (processor, errors) ->
      this.processorBackward(processor = processor, outputErrors = errors)
    }
  }

  /**
   * Execute the forward of the input of a new element to the output, instantiating a new dedicated feed-forward
   * processor.
   * This method must be used when the input layer is NOT a merge layer.
   *
   * @param input the input array
   *
   * @return the output array
   */
  private fun forwardProcessor(input: InputNDArrayType): DenseNDArray =
    this.processorsPool.getItem().let { this.usedProcessors.add(it); it.forward(input) }

  /**
   * Execute the forward of the input of a new element to the output, instantiating a new dedicated feed-forward
   * processor.
   * This method must be used when the input layer is a merge layer.
   *
   * @param input the input array
   *
   * @return the output array
   */
  private fun forwardProcessor(input: List<InputNDArrayType>): DenseNDArray =
    this.processorsPool.getItem().let { this.usedProcessors.add(it); it.forward(input) }

  /**
   * Reset the current batch.
   */
  private fun reset() {

    this.processorsPool.releaseAll()
    this.usedProcessors.clear()
    this.errorsAccumulator.clear()
  }

  /**
   * Execute the backward of a given [processor], given its output [outputErrors].
   *
   * @param processor a processor used for the current batch
   * @param outputErrors the output errors
   */
  private fun processorBackward(processor: FeedforwardNeuralProcessor<InputNDArrayType>, outputErrors: DenseNDArray) {

    processor.backward(outputErrors = outputErrors)

    this.errorsAccumulator.accumulate(processor.getParamsErrors(copy = false))
  }
}
