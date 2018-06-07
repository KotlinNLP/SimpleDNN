/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork.structure.feedforward

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.NetworkStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The FeedforwardNetworkStructure.
 *
 * @property layersConfiguration layers layersConfiguration
 * @property params the network parameters per layer
 */
class FeedforwardNetworkStructure<InputNDArrayType : NDArray<InputNDArrayType>>(
  layersConfiguration: List<LayerInterface>,
  params: NetworkParameters
) : NetworkStructure<InputNDArrayType>(layersConfiguration = layersConfiguration, params = params) {

  /**
   * LayerStructure factory used to concatenate two layers, given the input array (referenced from
   * the previous layer) and the output configuration.
   *
   * @param inputArray an AugmentedArray used as referenced input (to concatenate two layers)
   * @param outputConfiguration the configuration of the output interface of the layer
   * @param params the network parameters of the current layer
   *
   * @return a new LayerStructure
   */
  override fun <InputNDArrayType : NDArray<InputNDArrayType>> layerFactory(
    inputArray: AugmentedArray<InputNDArrayType>,
    outputConfiguration: LayerInterface,
    params: LayerParameters<*>,
    dropout: Double
  ): LayerStructure<InputNDArrayType> {

    require(outputConfiguration.connectionType!!.property == LayerType.Property.Feedforward) {
      "Layer connection of type ${outputConfiguration.connectionType} not allowed [only FeedForward connections]"
    }

    return LayerStructureFactory(
      inputArrays = listOf(inputArray),
      outputSize = outputConfiguration.size,
      params = params,
      activationFunction = outputConfiguration.activationFunction,
      connectionType = outputConfiguration.connectionType,
      dropout = dropout)
  }

  /**
   * Propagate the relevance from the output to the input of each layer, starting from the given distribution on
   * the outcomes.
   *
   * @param networkContributions the [NetworkParameters] in which to save the contributions during calculations
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   */
  fun propagateRelevance(networkContributions: NetworkParameters, relevantOutcomesDistribution: DistributionArray) {

    this.layers.last().setOutputRelevance(relevantOutcomesDistribution)

    for ((i, layer) in this.layers.withIndex().reversed()) { layer as FeedforwardLayerStructure
      this.curLayerIndex = i
      layer.setInputRelevance(layerContributions = networkContributions.paramsPerLayer[i])
    }
  }
}
