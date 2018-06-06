/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.RecurrentLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.NetworkStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The RecurrentNetworkStructure.
 *
 * @property layersConfiguration layers layersConfiguration
 * @property params the network parameters per layer
 * @property structureContextWindow the context window to get the previous and the next state of the structure
 */
class RecurrentNetworkStructure <InputNDArrayType : NDArray<InputNDArrayType>>(
  layersConfiguration: List<LayerConfiguration>,
  params: NetworkParameters,
  val structureContextWindow: StructureContextWindow<InputNDArrayType>
) : LayerContextWindow,
  NetworkStructure<InputNDArrayType>(layersConfiguration = layersConfiguration, params = params) {

  /**
   * A list of booleans indicating if the init hidden layers must be used in the next forward.
   */
  private var useInitHidden: List<Boolean> = this.layers.map { false }

  /**
   * The initial hidden layers from which to take the previous hidden if the method [setInitHidden] is called before a
   * forward.
   */
  private val initHiddenLayers: List<LayerStructure<*>> = this.buildLayers()

  /**
   * Set the initial hidden arrays of each layer. They will be used as previous hidden in the next forward.
   * Set [arrays] to null to don't use the initial hidden layers.
   *
   * @param arrays the list of initial hidden arrays (one per layer, can be null)
   */
  fun setInitHidden(arrays: List<DenseNDArray?>?) {
    require(arrays == null || arrays.size == this.layers.size) {
      "Incompatible init hidden arrays size (%d != %d).".format(arrays!!.size, this.layers.size)
    }

    if (arrays != null) {
      this.initHiddenLayers.zip(arrays).forEach { (layer, array) ->
        if (layer is RecurrentLayerStructure && array != null) layer.setInitHidden(array)
      }
    }

    this.useInitHidden = arrays?.map { it != null } ?: this.layers.map { false }
  }

  /**
   * Get the errors of the initial hidden arrays.
   * This method should be used only if initial hidden arrays has been set with the [setInitHidden] method.
   *
   * @return the errors of the initial hidden arrays (null if no init hidden is used for a certain layer)
   */
  fun getInitHiddenErrors(): List<DenseNDArray?> =
    this.useInitHidden.zip(this.initHiddenLayers).map { (useInitHidden, layer) ->
      if (useInitHidden && layer is RecurrentLayerStructure) layer.getInitHiddenErrors() else null
    }

  /**
   * @return the current layer in previous state
   */
  override fun getPrevStateLayer(): LayerStructure<*>? = if (this.useInitHidden[this.curLayerIndex]) {
    this.initHiddenLayers[this.curLayerIndex]
  } else {
    val prevStateStructure = this.structureContextWindow.getPrevStateStructure()
    prevStateStructure?.layers?.get(this.curLayerIndex)
  }

  /**
   * @return the current layer in next state
   */
  override fun getNextStateLayer(): LayerStructure<*>? {
    val nextStateStructure = this.structureContextWindow.getNextStateStructure()
    return nextStateStructure?.layers?.get(this.curLayerIndex)
  }

  /**
   * LayerStructure factory used to concatV two layers, given the input array (referenced from
   * the previous layer) and the output layersConfiguration.
   *
   * @param inputArray an AugmentedArray used as referenced input (to concatV two layers)
   * @param outputConfiguration the layersConfiguration of the output array
   * @param params the network parameters of the current layer
   * @param dropout the probability of dropout
   *
   * @return a new LayerStructure
   */
  override fun <InputNDArrayType : NDArray<InputNDArrayType>> layerFactory(
    inputArray: AugmentedArray<InputNDArrayType>,
    outputConfiguration: LayerConfiguration,
    params: LayerParameters<*>,
    dropout: Double
  ): LayerStructure<InputNDArrayType> = LayerStructureFactory(
    inputArray = inputArray,
    outputSize = outputConfiguration.size,
    params = params,
    activationFunction = outputConfiguration.activationFunction,
    connectionType = outputConfiguration.connectionType!!,
    dropout = dropout,
    contextWindow = this)
}
