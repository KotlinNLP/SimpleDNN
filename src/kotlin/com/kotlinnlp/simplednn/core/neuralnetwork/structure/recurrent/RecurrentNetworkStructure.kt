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
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.NetworkStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The RecurrentNetworkStructure.
 *
 * @param layersConfiguration layers layersConfiguration
 * @param params the network parameters per layer
 */
class RecurrentNetworkStructure <InputNDArrayType : NDArray<InputNDArrayType>>(
  layersConfiguration: List<LayerConfiguration>,
  params: NetworkParameters,
  val structureContextWindow: StructureContextWindow<InputNDArrayType>
) : LayerContextWindow,
    NetworkStructure<InputNDArrayType>(layersConfiguration = layersConfiguration, params = params) {

  /**
   *
   */
  override fun getPrevStateLayer(): LayerStructure<*>? {
    val prevStateStructure = this.structureContextWindow.getPrevStateStructure()
    return prevStateStructure?.layers?.get(this.curLayerIndex)
  }

  /**
   *
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
   *
   * @return a new LayerStructure
   */
  override fun <InputNDArrayType : NDArray<InputNDArrayType>> layerFactory(
    inputArray: AugmentedArray<InputNDArrayType>,
    outputConfiguration: LayerConfiguration,
    params: LayerParameters,
    dropout: Double): LayerStructure<InputNDArrayType> {

    return LayerStructureFactory<InputNDArrayType>(
      inputArray = inputArray,
      outputArray = AugmentedArray(outputConfiguration.size),
      params = params,
      activationFunction = outputConfiguration.activationFunction,
      connectionType = outputConfiguration.connectionType!!,
      dropout = dropout,
      contextWindow = this)
  }
}
