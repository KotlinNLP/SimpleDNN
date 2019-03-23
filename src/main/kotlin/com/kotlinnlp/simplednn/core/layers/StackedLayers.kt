/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The StackedLayers.
 *
 * @property layersConfiguration the layers configurations
 * @property paramsPerLayer the parameters per layer
 */
open class StackedLayers<InputNDArrayType : NDArray<InputNDArrayType>>(
  val layersConfiguration: List<LayerInterface>,
  val paramsPerLayer: List<LayerParameters<*>>
) : LayerContextWindow {

  /**
   * The list of layers generate from the [layersConfiguration].
   */
  val layers: List<Layer<*>> = this.buildLayers()

  /**
   * Used to track the loop over the layers during the forward and the backward
   */
  var curLayerIndex: Int = 0

  /**
   * The first layer
   */
  @Suppress("UNCHECKED_CAST")
  val inputLayer: Layer<InputNDArrayType> get() = layers.first() as Layer<InputNDArrayType>

  /**
   * The last layer
   */
  @Suppress("UNCHECKED_CAST")
  val outputLayer: Layer<DenseNDArray> get() = layers.last() as Layer<DenseNDArray>

  /**
   * Forward features.
   *
   * @param input the input to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: InputNDArrayType, useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(input)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Forward features, saving the contributions.
   *
   * @param input the input to forward from the input to the output
   * @param stackedLayersContributions the [StackedLayersParameters] in which to save the contributions of the input of each layer
   *                             in respect of the related output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: InputNDArrayType,
              stackedLayersContributions: StackedLayersParameters,
              useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(input)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(layerContributions = stackedLayersContributions.paramsPerLayer[i], useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Forward a list of features if the first layer is a Merge layer.
   *
   * @param input the input to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: List<InputNDArrayType>, useDropout: Boolean = false): DenseNDArray {

    require(this.inputLayer is MergeLayer<InputNDArrayType>) {
      "Cannot call the forward with multiple inputs if the first layer is not a Merge layer."
    }

    (this.inputLayer as MergeLayer<InputNDArrayType>).let {
      input.forEachIndexed { i, values -> it.setInput(index = i, values = values) }
    }

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Forward a list of features if the first layer is a Merge layer, saving the contributions.
   *
   * @param input the input to forward from the input to the output
   * @param stackedLayersContributions the [StackedLayersParameters] in which to save the contributions of the input of each layer
   *                             in respect of the related output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: List<InputNDArrayType>,
              stackedLayersContributions: StackedLayersParameters,
              useDropout: Boolean = false): DenseNDArray {

    require(this.inputLayer is MergeLayer<InputNDArrayType>) {
      "Cannot call the forward with multiple inputs if the first layer is not a Merge layer."
    }

    (this.inputLayer as MergeLayer<InputNDArrayType>).let {
      input.forEachIndexed { i, values -> it.setInput(index = i, values = values) }
    }

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(layerContributions = stackedLayersContributions.paramsPerLayer[i], useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Propagate the output error using the gradient descent algorithm
   *
   * @param outputErrors the errors to propagate from the output
   * @param propagateToInput whether to propagate the errors to the input
   *
   * @return the params errors
   */
  fun backward(outputErrors: DenseNDArray,
               propagateToInput: Boolean = false): ParamsErrorsList {

    this.outputLayer.setErrors(outputErrors)

    val paramsErrorsPerLayer = mutableListOf<List<ParamsArray.Errors<*>>>()

    for ((i, layer) in this.layers.withIndex().reversed()) {

      this.curLayerIndex = i

      paramsErrorsPerLayer.add(
        layer.backward(propagateToInput = (i > 0 || propagateToInput)))
    }

    return paramsErrorsPerLayer.flatten()
  }

  /**
   * Propagate the relevance from the output to the input of each layer, starting from the given distribution on
   * the outcomes.
   *
   * @param stackedLayersContributions the [StackedLayersParameters] in which to save the contributions during calculations
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   */
  fun propagateRelevance(stackedLayersContributions: StackedLayersParameters, relevantOutcomesDistribution: DistributionArray) {

    require(this.layers.all { it is FeedforwardLayer } ) {
      "The relevance propagation requires that all the layers must be feed-forward."
    }

    this.layers.last().setOutputRelevance(relevantOutcomesDistribution)

    for ((i, layer) in this.layers.withIndex().reversed()) { layer as FeedforwardLayer
      this.curLayerIndex = i
      layer.setInputRelevance(layerContributions = stackedLayersContributions.paramsPerLayer[i])
    }
  }

  /**
   * @return the current layer in previous state
   */
  override fun getPrevState(): Layer<*>? = null

  /**
   * @return the current layer in next state
   */
  override fun getNextState(): Layer<*>? = null

  /**
   * Set the given params errors collector [c] to all [layers].
   *
   * @param c a collector of params errors
   */
  fun setParamsErrorsCollector(c: ParamsErrorsCollector) { this.layers.forEach { it.setParamsErrorsCollector(c) } }

  /**
   * @return the list of layers generated from the [layersConfiguration]
   */
  protected fun buildLayers(): List<Layer<*>> = this.layersConfiguration.toLayers(
    paramsPerLayer = this.paramsPerLayer,
    contextWindow = this
  )
}
