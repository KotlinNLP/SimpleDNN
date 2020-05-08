/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.multihead

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayersPool
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A multi-head scaled-dot attention network.
 *
 * @property model the model parameters
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property useDropout whether to apply the attention dropout during the [forward]
 * @property id a unique ID
 */
class MultiHeadAttentionNetwork(
  val model: MultiHeadAttentionParameters,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * Dropout not available.
   */
  override val useDropout: Boolean = false

  /**
   * Contains the errors accumulated from the processors during the forward.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The scaled-dot attention layers used for the last forward.
   */
  private lateinit var attentionLayers: List<ScaledDotAttentionLayer>

  /**
   * The pool of merge layers of the multi-head attention outputs.
   */
  private val mergePool: ConcatFFLayersPool<DenseNDArray> =
    ConcatFFLayersPool(params = this.model.merge, inputType = LayerType.Input.Dense)

  /**
   * The merge layers of the attention outputs that have been used for the last forward.
   */
  private lateinit var mergeLayers: List<ConcatFFLayer<DenseNDArray>>

  /**
   * Forward an input sequence.
   *
   * @param input the input arrays
   *
   * @return the encoded arrays
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.initLayers(input)

    this.attentionLayers.forEach { it.forward(useDropout = this.useDropout) }

    return this.mergeLayers.mapIndexed { i, mergeLayer ->

      mergeLayer.inputArrays.zip(this.attentionLayers).forEach { (mergeInput, attentionLayer) ->
        mergeInput.assignValues(attentionLayer.outputArrays[i].values)
      }

      mergeLayer.forward()

      mergeLayer.outputArray.values
    }
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()

    this.mergeLayers.zip(outputErrors).forEachIndexed { i, (mergeLayer, errors) ->

      mergeLayer.setErrors(errors)
      this.errorsAccumulator.accumulate(mergeLayer.backward(propagateToInput = true))

      this.attentionLayers.zip(mergeLayer.getInputErrors(copy = false)).forEach { (attentionLayer, mergeErrors) ->
        attentionLayer.outputArrays[i].assignErrors(mergeErrors)
      }
    }

    this.attentionLayers.forEach { this.errorsAccumulator.accumulate(it.backward(this.propagateToInput), copy = false) }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {

    val inputErrors: List<DenseNDArray> = this.attentionLayers.first().inputArrays.map { it.errors.copy() }

    this.attentionLayers.takeLast(this.attentionLayers.size - 1).forEach {
      inputErrors.zip(it.inputArrays).forEach { (errors, input) -> errors.assignSum(input.errors) }
    }

    return inputErrors
  }

  /**
   * Initialize the layers.
   *
   * @param input the input sequence
   */
  private fun initLayers(input: List<DenseNDArray>) {

    this.attentionLayers = this.model.attention.map { params ->
      ScaledDotAttentionLayer(inputArrays = input.map { AugmentedArray(it) }, params = params)
    }

    this.mergeLayers = this.mergePool.releaseAndGetItems(input.size)
  }
}
