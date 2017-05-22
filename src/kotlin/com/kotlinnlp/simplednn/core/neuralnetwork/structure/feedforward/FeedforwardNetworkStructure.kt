/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.structure.feedforward

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.NetworkStructure

/**
 * The FeedforwardNetworkStructure.
 *
 * @param layersConfiguration layers layersConfiguration
 * @param params the network parameters per layer
 */
class FeedforwardNetworkStructure(
  layersConfiguration: List<LayerConfiguration>,
  params: NetworkParameters
) : NetworkStructure(layersConfiguration = layersConfiguration, params = params) {

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
  override fun layerFactory(inputArray: AugmentedArray,
                            outputConfiguration: LayerConfiguration,
                            params: LayerParameters,
                            dropout: Double): LayerStructure {

    require(outputConfiguration.connectionType == LayerType.Connection.Feedforward) {
      "Layer connection of type %s not allowed [only %s]".format(
        outputConfiguration.connectionType, LayerType.Connection.Feedforward)
    }

    return LayerStructureFactory(
      inputArray = inputArray,
      outputArray = AugmentedArray(outputConfiguration.size),
      params = params,
      activationFunction = outputConfiguration.activationFunction,
      connectionType = outputConfiguration.connectionType!!,
      dropout = dropout)
  }
}
