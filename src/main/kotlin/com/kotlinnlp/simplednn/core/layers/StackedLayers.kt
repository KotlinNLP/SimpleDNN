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
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * A structure of stacked layers, in which the output array of a layer references the input array of the following.
 * This permits to optimize the forward and backward operations without duplicating values.
 *
 * @property params the layers parameters
 */
internal open class StackedLayers<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: StackedLayersParameters
) {

  /**
   * The list of layers generate from the layers configuration contained inside the [params].
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
  val inputLayer: Layer<InputNDArrayType> get() = this.layers.first() as Layer<InputNDArrayType>

  /**
   * The last layer
   */
  @Suppress("UNCHECKED_CAST")
  val outputLayer: Layer<DenseNDArray> get() = this.layers.last() as Layer<DenseNDArray>

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
   * @param contributions the support in which to save the contributions of the input respect to the related output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: InputNDArrayType,
              contributions: StackedLayersParameters,
              useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(input)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(contributions = contributions.paramsPerLayer[i], useDropout = useDropout)
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
   * @param contributions the support in which to save the contributions of the input respect to the related output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: List<InputNDArrayType>,
              contributions: StackedLayersParameters,
              useDropout: Boolean = false): DenseNDArray {

    require(this.inputLayer is MergeLayer<InputNDArrayType>) {
      "Cannot call the forward with multiple inputs if the first layer is not a Merge layer."
    }

    (this.inputLayer as MergeLayer<InputNDArrayType>).let {
      input.forEachIndexed { i, values -> it.setInput(index = i, values = values) }
    }

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(contributions = contributions.paramsPerLayer[i], useDropout = useDropout)
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
   * @param layersContributions the contributions of the input respect to the related output
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   */
  fun propagateRelevance(layersContributions: StackedLayersParameters,
                         relevantOutcomesDistribution: DistributionArray) {

    require(this.layers.all { it is FeedforwardLayer } ) {
      "The relevance propagation requires that all the layers must be feed-forward."
    }

    this.layers.last().setOutputRelevance(relevantOutcomesDistribution)

    for ((i, layer) in this.layers.withIndex().reversed()) { layer as FeedforwardLayer
      this.curLayerIndex = i
      layer.setInputRelevance(contributions = layersContributions.paramsPerLayer[i])
    }
  }

  /**
   * Set the given params errors collector [c] to all [layers].
   *
   * @param c a collector of params errors
   */
  fun setParamsErrorsCollector(c: ParamsErrorsCollector) { this.layers.forEach { it.setParamsErrorsCollector(c) } }

  /**
   * Build a new list of layer.
   *
   * Layers are defined as a list [x, y, z] of [LayerInterface].
   * The resulting list of [Layer] consist in input-output pairs [x-y, y-z].
   *
   * @return list of layers where the output of a layer is the reference of the input of the next one
   */
  protected fun buildLayers(): List<Layer<*>> = this.params.layersConfiguration.let { config ->

    require(config.subList(1, config.size).all { it.type == LayerType.Input.Dense }) {
      "The last layers must be dense."
    }

    require(config.subList(2, config.size).all { it.connectionType!!.property != LayerType.Property.Merge }) {
      "Only the first layer can be a Merge layer."
    }

    var prevLayer: Layer<*>? = null

    return List(
      size = config.size - 1,
      init = { i ->

        val layer: Layer<*> = if (i == 0)
          LayerFactory(
            inputConfiguration = config[0],
            outputConfiguration = config[1],
            params = this.params.paramsPerLayer[0])
        else
          LayerFactory(
            inputArrays = listOf(prevLayer!!.outputArray),
            inputType = LayerType.Input.Dense,
            outputConfiguration = config[i + 1],
            params = this.params.paramsPerLayer[i],
            dropout = config[i].dropout)

        prevLayer = layer

        layer
      }
    )
  }
}
