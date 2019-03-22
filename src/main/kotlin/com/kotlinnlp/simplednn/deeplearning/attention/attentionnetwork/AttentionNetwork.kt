/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayersPool
import com.kotlinnlp.simplednn.core.attention.AttentionLayerStructure
import com.kotlinnlp.simplednn.core.layers.LayerType
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
 * @property dropout the probability of dropout (default 0.0) when generating the attention arrays for the Attention
 *                   Layer. If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific [AttentionNetwork]
 */
class AttentionNetwork<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: AttentionNetworkParameters,
  val inputType: LayerType.Input,
  val dropout: Double = 0.0,
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * The accumulator of errors of the transform layer parameters.
   */
  private val transformParamsErrorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The transform layer which creates an attention array from each array of an input sequence.
   */
  private lateinit var transformLayers: List<FeedforwardLayer<InputNDArrayType>>

  /**
   * The transform layers pool.
   */
  private var transformLayersPool = FeedforwardLayersPool<InputNDArrayType>(
    params = this.model.transformParams,
    inputType = this.inputType,
    activationFunction = Tanh(),
    dropout = this.dropout
  )

  /**
   * The Attention Layer of input.
   */
  private lateinit var attentionLayer: AttentionLayerStructure<InputNDArrayType>

  /**
   * A boolean indicating if attention arrays automatically generated have been used during the last forward.
   */
  private var internalAttentionArraysUsed: Boolean = false

  /**
   * Forward an input sequence, building the attention arrays automatically, encoding the [inputSequence] through a
   * Feedforward Neural Layer.
   *
   * @param inputSequence the list of input arrays
   * @param useDropout whether to apply the dropout to generate the attention arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(inputSequence: List<AugmentedArray<InputNDArrayType>>,
              useDropout: Boolean = false): DenseNDArray {

    this.internalAttentionArraysUsed = true

    this.setInputSequence(inputSequence = inputSequence, useDropout = useDropout)
    this.attentionLayer.forward()

    return this.attentionLayer.outputArray.values
  }

  /**
   * Forward an input sequence, given its related attention arrays.
   *
   * @param inputSequence the list of input arrays
   * @param attentionSequence the list of attention arrays
   *
   * @return the output [DenseNDArray]
   */
  fun forward(inputSequence: List<AugmentedArray<InputNDArrayType>>,
              attentionSequence: List<DenseNDArray>): DenseNDArray {

    this.internalAttentionArraysUsed = false

    this.attentionLayer = AttentionLayerStructure(
      inputSequence = inputSequence,
      attentionSequence = attentionSequence,
      params = this.model.attentionParams)

    this.attentionLayer.forward()

    return this.attentionLayer.outputArray.values
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the errors to propagate from the output
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrors: DenseNDArray,
               propagateToInput: Boolean = false): ParamsErrorsList {

    val paramsErrors = mutableListOf<ParamsErrorsList>()

    paramsErrors.add(
      this.backwardAttentionLayer(
        outputErrors = outputErrors,
        propagateToInput = propagateToInput))

    if (this.internalAttentionArraysUsed) {

      // WARNING: call it after the backward of the attention layer
      paramsErrors.add(
        this.backwardTransformLayers(propagateToInput))

      if (propagateToInput) {
        this.addTransformErrorsToInput()
      }
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
  fun getInputErrors(): List<DenseNDArray> = this.attentionLayer.inputSequence.map { it.errors }

  /**
   * @return the errors of the attention arrays
   */
  fun getAttentionErrors(): List<DenseNDArray> = this.attentionLayer.getAttentionErrors()

  /**
   * @param copy a Boolean indicating whether the returned importance score must be a copy or a reference
   *
   * @return the importance score of each array of input in a [DenseNDArray]
   */
  fun getImportanceScore(copy: Boolean = true): DenseNDArray =
    if (copy)
      this.attentionLayer.importanceScore.values.copy()
    else
      this.attentionLayer.importanceScore.values

  /**
   * Set the input sequence.
   *
   * @param inputSequence the list of arrays of input
   * @param useDropout whether to apply the dropout to generate the attention arrays
   */
  private fun setInputSequence(inputSequence: List<AugmentedArray<InputNDArrayType>>,
                               useDropout: Boolean = false) {

    this.attentionLayer = AttentionLayerStructure(
      inputSequence = inputSequence,
      attentionSequence = this.buildAttentionSequence(inputSequence = inputSequence, useDropout = useDropout),
      params = this.model.attentionParams
    )
  }

  /**
   * @param inputSequence the list of arrays of input
   * @param useDropout whether to apply the dropout
   *
   * @return the list of attention arrays associated to each array of the [inputSequence]
   */
  private fun buildAttentionSequence(
    inputSequence: List<AugmentedArray<InputNDArrayType>>,
    useDropout: Boolean
  ): List<DenseNDArray> {

    this.transformLayersPool.releaseAll()

    this.transformLayers = List(size = inputSequence.size, init = { this.transformLayersPool.getItem() })

    return inputSequence.mapIndexed { i, inputArray ->

      val layer = this.transformLayers[i]

      layer.setInput(inputArray.values)
      layer.forward(useDropout = useDropout)

      layer.outputArray.values
    }
  }

  /**
   * Attention Layer backward.
   *
   * @param outputErrors the errors to propagate from the output
   * @param paramsErrors the structure in which to save the errors of the parameters
   * @param propagateToInput whether to propagate the errors to the input
   */
  private fun backwardAttentionLayer(outputErrors: DenseNDArray,
                                     propagateToInput: Boolean = false): ParamsErrorsList {

    this.attentionLayer.setOutputErrors(outputErrors)
    return this.attentionLayer.backward(propagateToInput = propagateToInput)
  }

  /**
   * Transform Layers backward.
   *
   * @param propagateToInput whether to propagate the errors to the input
   */
  private fun backwardTransformLayers(propagateToInput: Boolean = false): ParamsErrorsList {

    val attentionErrors: List<DenseNDArray> = this.getAttentionErrors()

    // Accumulate errors into the accumulator
    this.transformLayers.forEachIndexed { i, layer ->
      layer.setErrors(attentionErrors[i])
      this.transformParamsErrorsAccumulator.accumulate(layer.backward(propagateToInput))
    }

    this.transformParamsErrorsAccumulator.averageErrors()

    val accumulatedErrors = this.transformParamsErrorsAccumulator.getParamsErrors()

    this.transformParamsErrorsAccumulator.clear()

    return accumulatedErrors
  }

  /**
   * Add the input errors of the transform layer to each input array.
   */
  private fun addTransformErrorsToInput() {

    this.attentionLayer.inputSequence.forEachIndexed { i, inputArray ->
      inputArray.errors.assignSum(this.transformLayers[i].inputArray.errors)
    }
  }
}
