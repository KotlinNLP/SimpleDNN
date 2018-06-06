/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.RecurrentNetworkStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.StructureContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import core.layers.structure.utils.FeedforwardLayerStructureUtils
import core.layers.structure.utils.SimpleRecurrentLayerStructureUtils

/**
 *
 */
object RecurrentNetworkStructureUtils {

  /**
   *
   */
  fun buildParams(layersConfiguration: List<LayerInterface>): NetworkParameters {

    val params = NetworkParameters(layersConfiguration)
    val inputParams = (params.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
    val outputParams = (params.paramsPerLayer[1] as FeedforwardLayerParameters)
    val recurrentParams = SimpleRecurrentLayerStructureUtils.buildParams()
    val feedforwardParams = FeedforwardLayerStructureUtils.getParams53()

    inputParams.unit.weights.values.assignValues(recurrentParams.unit.weights.values)
    inputParams.unit.biases.values.assignValues(recurrentParams.unit.biases.values)
    inputParams.unit.recurrentWeights.values.assignValues(recurrentParams.unit.recurrentWeights.values)
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
      LayerInterface(size = 4),
      LayerInterface(size = 5, activationFunction = Tanh(), connectionType = LayerType.Connection.SimpleRecurrent),
      LayerInterface(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward)
    ).toList()

    return RecurrentNetworkStructure(
      layersConfiguration = layersConfiguration,
      params = this.buildParams(layersConfiguration),
      structureContextWindow = structureContextWindow)
  }
}
