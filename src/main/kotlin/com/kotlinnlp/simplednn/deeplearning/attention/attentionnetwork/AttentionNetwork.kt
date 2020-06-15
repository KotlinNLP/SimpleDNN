/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.attention.AttentionLayer
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * The Attention Network which classifies an input sequence using an Attention Layer and a Feedforward Layer as
 * transform layer.
 *
 * @property model the parameters of the model of the network
 * @property inputType the type of the input arrays
 * @param dropout the probability of dropout when generating the attention arrays (default 0.0)
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @property id an identification number useful to track a specific [AttentionNetwork]
 */
class AttentionNetwork<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: AttentionNetworkParameters,
  val inputType: LayerType.Input,
  dropout: Double = 0.0,
  private val propagateToInput: Boolean,
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * The accumulator of errors of the transform layer parameters.
   */
  private val transformParamsErrorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The transform layer which creates an attention array from each array of an input sequence.
   */
  private val transformProcessor: BatchFeedforwardProcessor<InputNDArrayType> =
    BatchFeedforwardProcessor(model = this.model.transform, dropout = dropout, propagateToInput = this.propagateToInput)

  /**
   * The Attention Layer of input.
   */
  private lateinit var attentionLayer: AttentionLayer<InputNDArrayType>

  /**
   * A boolean indicating if attention arrays automatically generated have been used during the last forward.
   */
  private var internalAttentionArraysUsed: Boolean = false

  /**
   * Forward an input sequence, building the attention arrays automatically, encoding the [inputSequence] through a
   * Feedforward Neural Layer.
   *
   * @param inputSequence the input arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(inputSequence: List<InputNDArrayType>): DenseNDArray {

    this.internalAttentionArraysUsed = true

    this.setInputSequence(inputSequence)
    this.attentionLayer.forward()

    return this.attentionLayer.outputArray.values
  }

  /**
   * Forward an input sequence, given its related attention arrays.
   *
   * @param inputSequence the input arrays
   * @param attentionSequence the attention arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(inputSequence: List<InputNDArrayType>, attentionSequence: List<DenseNDArray>): DenseNDArray {

    this.internalAttentionArraysUsed = false

    this.attentionLayer = AttentionLayer(
      inputArrays = inputSequence.map { AugmentedArray(it) },
      inputType = this.inputType,
      attentionArrays = attentionSequence.map { AugmentedArray(it) },
      params = this.model.attention)

    this.attentionLayer.forward()

    return this.attentionLayer.outputArray.values
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the errors to propagate from the output
   *
   * @return the params errors
   */
  fun backward(outputErrors: DenseNDArray): ParamsErrorsList {

    val paramsErrors = mutableListOf<ParamsErrorsList>()

    paramsErrors.add(this.backwardAttentionLayer(outputErrors = outputErrors, propagateToInput = propagateToInput))

    if (this.internalAttentionArraysUsed) {

      // WARNING: call it after the backward of the attention layer
      paramsErrors.add(this.backwardTransformLayers())

      if (this.propagateToInput)
        this.addTransformErrorsToInput()
    }

    return paramsErrors.flatten()
  }

  /**
   * @param copy a Boolean indicating whether the returned value must be a copy or a reference
   *
   * @return the output of the last forward() in a [DenseNDArray]
   */
  fun getOutput(copy: Boolean = true): DenseNDArray =
    if (copy)
      this.attentionLayer.outputArray.values.copy()
    else
      this.attentionLayer.outputArray.values

  /**
   * @return the errors of the arrays of input
   */
  fun getInputErrors(): List<DenseNDArray> = this.attentionLayer.inputArrays.map { it.errors }

  /**
   * @return the errors of the attention arrays
   */
  fun getAttentionErrors(): List<DenseNDArray> = this.attentionLayer.attentionArrays.map { it.errors }

  /**
   * @param copy a Boolean indicating whether the returned importance score must be a copy or a reference
   *
   * @return the importance score of each array of input in a [DenseNDArray]
   */
  fun getImportanceScore(copy: Boolean = true): DenseNDArray =
    if (copy)
      this.attentionLayer.attentionScores.values.copy()
    else
      this.attentionLayer.attentionScores.values

  /**
   * Set the input sequence.
   *
   * @param inputSequence the list of arrays of input
   */
  private fun setInputSequence(inputSequence: List<InputNDArrayType>) {

    val attentionSequence: List<DenseNDArray> = this.transformProcessor.forward(inputSequence)

    this.attentionLayer = AttentionLayer(
      inputArrays = inputSequence.map { AugmentedArray(it) },
      inputType = this.inputType,
      attentionArrays = attentionSequence.map { AugmentedArray(it) },
      params = this.model.attention)
  }

  /**
   * Attention Layer backward.
   *
   * @param outputErrors the errors to propagate from the output
   * @param propagateToInput whether to propagate the errors to the input
   */
  private fun backwardAttentionLayer(outputErrors: DenseNDArray,
                                     propagateToInput: Boolean = false): ParamsErrorsList {

    this.attentionLayer.outputArray.assignErrors(outputErrors)
    return this.attentionLayer.backward(propagateToInput = propagateToInput)
  }

  /**
   * Transform Layers backward.
   */
  private fun backwardTransformLayers(): ParamsErrorsList {

    val attentionErrors: List<DenseNDArray> = this.getAttentionErrors()

    this.transformProcessor.backward(attentionErrors)
    this.transformParamsErrorsAccumulator.accumulate(this.transformProcessor.getParamsErrors(copy = false))

    return this.transformParamsErrorsAccumulator.getParamsErrors(copy = true).also {
      this.transformParamsErrorsAccumulator.clear()
    }
  }

  /**
   * Add the input errors of the transform layer to each input array.
   */
  private fun addTransformErrorsToInput() {

    val transformInputErrors: List<DenseNDArray> = this.transformProcessor.getInputErrors(copy = false)

    this.attentionLayer.inputArrays.zip(transformInputErrors).forEach { (inputArray, transformErrors) ->
      inputArray.errors.assignSum(transformErrors)
    }
  }
}
