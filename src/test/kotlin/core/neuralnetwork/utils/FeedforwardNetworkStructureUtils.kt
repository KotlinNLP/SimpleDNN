/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork.utils

import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.types.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils


/**
 *
 */
object FeedforwardNetworkStructureUtils {

  fun buildParams(layersConfiguration: List<LayerInterface>): NetworkParameters {

    val params = NetworkParameters(layersConfiguration)
    val inputParams = (params.paramsPerLayer[0] as FeedforwardLayerParameters)
    val outputParams = (params.paramsPerLayer[1] as FeedforwardLayerParameters)

    inputParams.unit.weights.values.assignValues(FeedforwardLayerStructureUtils.getParams45().unit.weights.values)
    inputParams.unit.biases.values.assignValues(FeedforwardLayerStructureUtils.getParams45().unit.biases.values)
    outputParams.unit.weights.values.assignValues(FeedforwardLayerStructureUtils.getParams53().unit.weights.values)
    outputParams.unit.biases.values.assignValues(FeedforwardLayerStructureUtils.getParams53().unit.biases.values)

    return params
  }
}
