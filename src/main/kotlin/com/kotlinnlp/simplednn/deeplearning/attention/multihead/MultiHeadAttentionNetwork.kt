/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.multihead

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A multi-head scaled-dot attention network.
 *
 * @property model the model parameters
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
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
   * Contains the errors accumulated from the processors during the forward.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The scaled-dot attention layers used for the last forward.
   */
  private lateinit var attentionLayers: List<ScaledDotAttentionLayer>

  /**
   * The processor that merges the attention outputs.
   */
  private val mergeProcessor: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.model.merge, propagateToInput = true)

  /**
   * Forward an input sequence.
   *
   * @param input the input arrays
   *
   * @return the encoded arrays
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.attentionLayers = this.model.attention.map { params ->
      ScaledDotAttentionLayer(inputArrays = input.map { AugmentedArray(it) }, params = params)
    }

    val attentionOutputs: List<List<DenseNDArray>> = this.attentionLayers.map { layer ->
      layer.forward()
      layer.outputArrays.map { it.values }
    }

    return this.mergeProcessor.forward(attentionOutputs.foldUp().toTypedArray())
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()

    this.mergeProcessor.backward(outputErrors)
    this.errorsAccumulator.accumulate(this.mergeProcessor.getParamsErrors(copy = false))

    val attentionErrors: List<List<DenseNDArray>> = this.mergeProcessor.getInputsErrors(copy = false).foldUp()

    this.attentionLayers.zip(attentionErrors).forEach { (layer, attHeadErrors) ->
      layer.outputArrays.zip(attHeadErrors).forEach { (array, errors) -> array.assignErrors(errors) }
      this.errorsAccumulator.accumulate(layer.backward(this.propagateToInput), copy = false)
    }
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
   * Fold up these nested lists inverting the outer with the inner lists.
   * All the inner lists must contain the same number of elements.
   *
   * If the outer list contains M inner lists and, in turn, each of them contains N elements, then a list
   * with N inner lists will be given in output, each with M elements.
   *
   * @return the outer list folded up with the inner lists
   */
  private fun <T> List<List<T>>.foldUp(): List<List<T>> {

    val outer = this
    val ret: List<MutableList<T>> = List(outer.first().size) { mutableListOf<T>() }

    outer.forEach { inner -> // i .. M
      inner.forEachIndexed { j, elm -> // j .. N
        ret[j].add(elm)
      }
    }

    return ret.map { it.toList() }
  }
}
