/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralnetwork.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.RecurrentNetworkStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.StructureContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import layers.structure.utils.FeedforwardLayerStructureUtils
import layers.structure.utils.SimpleRecurrentLayerStructureUtils

/**
 *
 */
object RecurrentNetworkStructureUtils {

  /**
   *
   */
  fun buildParams(layersConfiguration: List<LayerConfiguration>): NetworkParameters {

    val params = NetworkParameters(layersConfiguration)
    val inputParams = (params.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
    val outputParams = (params.paramsPerLayer[1] as FeedforwardLayerParameters)
    val recurrentParams = SimpleRecurrentLayerStructureUtils.buildParams()
    val feedforwardParams = FeedforwardLayerStructureUtils.getParams53()

    inputParams.weights.values.assignValues(recurrentParams.weights.values)
    inputParams.biases.values.assignValues(recurrentParams.biases.values)
    inputParams.recurrentWeights.values.assignValues(recurrentParams.recurrentWeights.values)
    outputParams.unit.weights.values.assignValues(feedforwardParams.unit.weights.values)
    outputParams.unit.biases.values.assignValues(feedforwardParams.unit.biases.values)

    return params
  }

  /**
   *
   */
  fun buildStructure(structureContextWindow: StructureContextWindow<DenseNDArray>):
    RecurrentNetworkStructure<DenseNDArray> {

    val layersConfiguration = arrayOf(
      LayerConfiguration(size = 4),
      LayerConfiguration(size = 5, activationFunction = Tanh(), connectionType = LayerType.Connection.SimpleRecurrent),
      LayerConfiguration(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward)
    ).toList()

    return RecurrentNetworkStructure(
      layersConfiguration = layersConfiguration,
      params = this.buildParams(layersConfiguration),
      structureContextWindow = structureContextWindow)
  }
}
