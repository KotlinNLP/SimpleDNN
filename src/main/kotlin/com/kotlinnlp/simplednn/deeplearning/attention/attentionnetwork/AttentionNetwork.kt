/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayersPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.attention.AttentionParameters
import com.kotlinnlp.simplednn.core.attention.AttentionLayerStructure
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
  private val transformParamsErrorsAccumulator = ParamsErrorsAccumulator<FeedforwardLayerParameters>()

  /**
   * The transform layer which creates an attention array from each array of an input sequence.
   */
  private lateinit var transformLayers: List<FeedforwardLayerStructure<InputNDArrayType>>

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
   * @param paramsErrors the structure in which to save the errors of the parameters
   * @param propagateToInput whether to propagate the errors to the input
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors of the transform layers(ignored if null, the default)
   */
  fun backward(outputErrors: DenseNDArray,
               paramsErrors: AttentionNetworkParameters,
               propagateToInput: Boolean = false,
               mePropK: Double? = null) {

    this.backwardAttentionLayer(
      outputErrors = outputErrors,
      paramsErrors = paramsErrors.attentionParams,
      propagateToInput = propagateToInput)

    if (this.internalAttentionArraysUsed) {

      // WARNING: call it after the backward of the attention layer
      this.backwardTransformLayers(
        paramsErrors = paramsErrors.transformParams,
        propagateToInput = propagateToInput,
        mePropK = mePropK)

      if (propagateToInput) {
        this.addTransformErrorsToInput()
      }
    }
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
      this.attentionLayer.importanceScore.copy()
    else
      this.attentionLayer.importanceScore

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
                                     paramsErrors: AttentionParameters,
                                     propagateToInput: Boolean = false) {

    this.attentionLayer.setOutputErrors(outputErrors)
    this.attentionLayer.backward(paramsErrors = paramsErrors, propagateToInput = propagateToInput)
  }

  /**
   * Transform Layers backward.
   *
   * @param paramsErrors the structure in which to save the errors of the parameters
   * @param propagateToInput whether to propagate the errors to the input
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  private fun backwardTransformLayers(paramsErrors: FeedforwardLayerParameters,
                                      propagateToInput: Boolean = false,
                                      mePropK: Double?) {

    val attentionErrors: List<DenseNDArray> = this.getAttentionErrors()

    // Accumulate errors into the accumulator
    this.transformLayers.forEachIndexed { i, layer ->
      layer.setErrors(attentionErrors[i])
      layer.backward(paramsErrors = paramsErrors, propagateToInput = propagateToInput, mePropK = mePropK)
      this.transformParamsErrorsAccumulator.accumulate(paramsErrors)
    }

    this.transformParamsErrorsAccumulator.averageErrors()

    val accumulatedErrors: FeedforwardLayerParameters = this.transformParamsErrorsAccumulator.getParamsErrors()
    paramsErrors.zip(accumulatedErrors).forEach { (a, b) -> a.values.assignValues(b.values) }

    this.transformParamsErrorsAccumulator.reset()
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
