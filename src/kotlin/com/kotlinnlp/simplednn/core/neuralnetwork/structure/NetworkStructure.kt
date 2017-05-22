/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.structure

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 * The NetworkStructure.
 *
 * @param layersConfiguration layers layersConfiguration
 * @param params the network parameters per layer
 */
abstract class NetworkStructure(
  val layersConfiguration: List<LayerConfiguration>,
  val params: NetworkParameters) {

  /**
   * Contains the layers of the neural network
   */
  val layers: Array<LayerStructure>

  /**
   * Used to track the loop over the layers during the forward and the backward
   */
  var curLayerIndex: Int = 0

  /**
   * The first layer
   */
  val inputLayer: LayerStructure get() = layers.first()

  /**
   * The last layer
   */
  val outputLayer: LayerStructure get() = layers.last()

  /**
   * In the layersConfiguration layers are defined as a list [x, y, z].
   * The structure contains layers as input-output pairs [x-y, y-z].
   * The output of a layer is a reference of the input of the next layer.
   */
  init {

    val initLayer = arrayOfNulls<LayerStructure>(layersConfiguration.size - 1)

    initLayer[0] = this.layerFactory(
      inputConfiguration = layersConfiguration[0],
      outputConfiguration = layersConfiguration[1],
      params = this.params.paramsPerLayer[0]
    )

    for (i in 1 until layersConfiguration.size - 1) {
      initLayer[i] = this.layerFactory(
        inputArray = initLayer[i - 1]!!.outputArray,
        outputConfiguration = layersConfiguration[i + 1],
        params = this.params.paramsPerLayer[i],
        dropout = layersConfiguration[i].dropout
      )
    }

    this.layers = initLayer.requireNoNulls()
  }

  /**
   * Forward
   *
   * @param features features to forward
   * @param useDropout whether to use the dropout
   */
  fun forward(features: NDArray, useDropout: Boolean = false): NDArray {

    this.inputLayer.setInput(features)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Propagate the output error using the gradient descent algorithm
   *
   * @param outputErrors errors to propagate
   * @param paramsErrors the error on the network parameters
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrors: NDArray,
               paramsErrors: NetworkParameters,
               propagateToInput: Boolean = false) {

    this.outputLayer.setErrors(outputErrors)

    for ((i, layer) in this.layers.withIndex().reversed()) {
      this.curLayerIndex = i
      layer.backward(paramsErrors = paramsErrors.paramsPerLayer[i], propagateToInput = (i > 0 || propagateToInput))
    }
  }

  /**
   * Base layer factory, given input and output configurations.
   *
   * @param inputConfiguration layers layersConfiguration of the input array
   * @param outputConfiguration the layersConfiguration of the output array
   * @param params the network parameters of the current layer
   *
   * @return a new LayerStructure
   */
  private fun layerFactory(inputConfiguration: LayerConfiguration,
                           outputConfiguration: LayerConfiguration,
                           params: LayerParameters): LayerStructure {

    return this.layerFactory(
      inputArray = AugmentedArray(inputConfiguration.size),
      outputConfiguration = outputConfiguration,
      params = params,
      dropout = inputConfiguration.dropout)
  }

  /**
   * LayerStructure factory used to concatV two layers, given the input array (referenced from
   * the previous layer) and the output layersConfiguration.
   *
   * @param inputArray an AugmentedArray used as referenced input (to concatV two layers)
   * @param outputConfiguration the layersConfiguration of the output array
   * @param params the network parameters of the current layer
   *
   * @return a new LayerStructure
   */
  abstract protected fun layerFactory(inputArray: AugmentedArray,
                                      outputConfiguration: LayerConfiguration,
                                      params: LayerParameters,
                                      dropout: Double): LayerStructure
}
