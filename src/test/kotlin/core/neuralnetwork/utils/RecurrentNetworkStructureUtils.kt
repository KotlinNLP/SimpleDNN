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
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.RecurrentStackedLayers
import com.kotlinnlp.simplednn.core.layers.StatesWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils
import core.layers.recurrent.simple.SimpleRecurrentLayerStructureUtils

/**
 *
 */
internal object RecurrentNetworkStructureUtils {

  /**
   *
   */
  fun buildParams(layersConfiguration: List<LayerInterface>) = StackedLayersParameters(layersConfiguration).apply {

    getLayerParams<SimpleRecurrentLayerParameters>(0).apply {

      val recurrentParams = SimpleRecurrentLayerStructureUtils.buildParams()

      unit.weights.values.assignValues(recurrentParams.unit.weights.values)
      unit.biases.values.assignValues(recurrentParams.unit.biases.values)
      unit.recurrentWeights.values.assignValues(recurrentParams.unit.recurrentWeights.values)
    }

    getLayerParams<FeedforwardLayerParameters>(1).apply {

      val feedforwardParams = FeedforwardLayerStructureUtils.getParams53()

      unit.weights.values.assignValues(feedforwardParams.unit.weights.values)
      unit.biases.values.assignValues(feedforwardParams.unit.biases.values)
    }
  }

  /**
   *
   */
  fun buildLayers(statesWindow: StatesWindow<DenseNDArray>): RecurrentStackedLayers<DenseNDArray> {

    val layersConfiguration = listOf(
      LayerInterface(size = 4),
      LayerInterface(size = 5, activationFunction = Tanh, connectionType = LayerType.Connection.SimpleRecurrent),
      LayerInterface(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward))

    return RecurrentStackedLayers(
      params = this.buildParams(layersConfiguration),
      statesWindow = statesWindow,
      dropout = 0.0)
  }
}
