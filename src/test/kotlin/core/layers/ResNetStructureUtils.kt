/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers

import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils

object ResNetStructureUtils {
  /**
   *
   */
  fun getParams53(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

    params.unit.weights.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.8, -0.8, 0.9, -1.0, -0.1),
            doubleArrayOf(0.9, 0.6, 0.7, 0.6, 0.6),
            doubleArrayOf(-0.1, 0.3, 0.3, 0.7, 0.3)
        )))

    params.unit.biases.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1, 0.2)))

    return params
  }

  /**
   *
   */
  fun getParams43(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 4, outputSize = 3)

    params.unit.weights.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.2, 0.8, 0.9, -1.0),
            doubleArrayOf(0.9, -0.6, -0.4, 0.6),
            doubleArrayOf(-0.1, 0.0, 0.3, -0.5)
        )))

    params.unit.biases.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1, 0.2)))

    return params
  }

  /**
   *
   */
  fun buildParams(layersConfiguration: List<LayerInterface>): StackedLayersParameters {

    val params = StackedLayersParameters(layersConfiguration)
    val inputParams = (params.paramsPerLayer[0] as FeedforwardLayerParameters)
    val outputParams = (params.paramsPerLayer[1] as FeedforwardLayerParameters)

    inputParams.unit.weights.values.assignValues(FeedforwardLayerStructureUtils.getParams45().unit.weights.values)
    inputParams.unit.biases.values.assignValues(FeedforwardLayerStructureUtils.getParams45().unit.biases.values)
    outputParams.unit.weights.values.assignValues(this.getParams53().unit.weights.values)
    outputParams.unit.biases.values.assignValues(this.getParams53().unit.biases.values)

    return params
  }
}