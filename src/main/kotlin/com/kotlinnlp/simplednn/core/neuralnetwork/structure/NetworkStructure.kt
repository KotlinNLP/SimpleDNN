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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The NetworkStructure.
 *
 * @property layersConfiguration layers layersConfiguration
 * @property params the network parameters per layer
 */
abstract class NetworkStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  val layersConfiguration: List<LayerConfiguration>,
  val params: NetworkParameters) {

  /**
   * Contains the layers of the neural network
   */
  val layers: Array<LayerStructure<*>>

  /**
   * Used to track the loop over the layers during the forward and the backward
   */
  var curLayerIndex: Int = 0

  /**
   * The first layer
   */
  @Suppress("UNCHECKED_CAST")
  val inputLayer: LayerStructure<InputNDArrayType> get() = layers.first() as LayerStructure<InputNDArrayType>

  /**
   * The last layer
   */
  @Suppress("UNCHECKED_CAST")
  val outputLayer: LayerStructure<DenseNDArray> get() = layers.last() as LayerStructure<DenseNDArray>

  /**
   * In the layersConfiguration layers are defined as a list [x, y, z].
   * The structure contains layers as input-output pairs [x-y, y-z].
   * The output of a layer is a reference of the input of the next layer.
   */
  init {
    require(layersConfiguration.subList(1, layersConfiguration.lastIndex).all {
      it.inputType == LayerType.Input.Dense
    })

    val initLayers = arrayOfNulls<LayerStructure<*>>(layersConfiguration.size - 1)

    initLayers[0] = this.layerFactory(
      inputConfiguration = layersConfiguration[0],
      outputConfiguration = layersConfiguration[1],
      params = this.params.paramsPerLayer[0]
    )

    for (i in 1 until layersConfiguration.size - 1) {
      initLayers[i] = this.layerFactory(
        inputArray = initLayers[i - 1]!!.outputArray,
        outputConfiguration = layersConfiguration[i + 1],
        params = this.params.paramsPerLayer[i],
        dropout = layersConfiguration[i].dropout
      )
    }

    this.layers = initLayers.requireNoNulls()
  }

  /**
   * Forward features.
   *
   * @param features the features to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(features: InputNDArrayType, useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(features)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Forward features, saving the contributions.
   *
   * @param features the features to forward from the input to the output
   * @param networkContributions the [NetworkParameters] in which to save the contributions of the input of each layer
   *                             in respect of the related output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(features: InputNDArrayType,
              networkContributions: NetworkParameters,
              useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(features)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(layerContributions = networkContributions.paramsPerLayer[i], useDropout = useDropout)
    }

    return this.outputLayer.outputArray.values
  }

  /**
   * Propagate the output error using the gradient descent algorithm
   *
   * @param outputErrors the errors to propagate from the output
   * @param paramsErrors the structure in which to save the errors of the parameters
   * @param propagateToInput whether to propagate the errors to the input
   * @param mePropK a list of k factors (one per layer) of the 'meProp' algorithm to propagate from the k (in
   *                percentage) output nodes with the top errors of each layer (the list and each element can be null)
   */
  fun backward(outputErrors: DenseNDArray,
               paramsErrors: NetworkParameters,
               propagateToInput: Boolean = false,
               mePropK: List<Double?>?) {

    require(mePropK == null || mePropK.size == this.layers.size) {
      "Invalid size of the list of mePropK factors: needed one factor per layer."
    }

    this.outputLayer.setErrors(outputErrors)

    for ((i, layer) in this.layers.withIndex().reversed()) {
      this.curLayerIndex = i
      layer.backward(
        paramsErrors = paramsErrors.paramsPerLayer[i],
        propagateToInput = (i > 0 || propagateToInput),
        mePropK = mePropK?.get(i))
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
  private fun layerFactory(
    inputConfiguration: LayerConfiguration,
    outputConfiguration: LayerConfiguration,
    params: LayerParameters<*>): LayerStructure<*> {

    require(outputConfiguration.connectionType != null) {
      "Output layer configurations must have a not null connectionType"
    }

    return when (inputConfiguration.inputType) {

      LayerType.Input.Dense -> this.layerFactory(
        inputArray = AugmentedArray<DenseNDArray>(size = inputConfiguration.size),
        outputConfiguration = outputConfiguration,
        params = params,
        dropout = inputConfiguration.dropout)

      LayerType.Input.Sparse -> this.layerFactory(
        inputArray = AugmentedArray<SparseNDArray>(size = inputConfiguration.size),
        outputConfiguration = outputConfiguration,
        params = params,
        dropout = inputConfiguration.dropout)

      LayerType.Input.SparseBinary -> this.layerFactory(
        inputArray = AugmentedArray<SparseBinaryNDArray>(size = inputConfiguration.size),
        outputConfiguration = outputConfiguration,
        params = params,
        dropout = inputConfiguration.dropout)
    }
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
  abstract protected fun <InputNDArrayType : NDArray<InputNDArrayType>> layerFactory(
    inputArray: AugmentedArray<InputNDArrayType>,
    outputConfiguration: LayerConfiguration,
    params: LayerParameters<*>,
    dropout: Double): LayerStructure<InputNDArrayType>
}
