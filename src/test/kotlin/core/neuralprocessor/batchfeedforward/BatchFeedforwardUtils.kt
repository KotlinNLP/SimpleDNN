/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object BatchFeedforwardUtils {

  /**
   *
   */
  fun buildInputBatch(): Array<DenseNDArray> = arrayOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, -0.8)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, -0.9, 0.6)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.3, -0.6))
  )

  /**
   *
   */
  fun buildOutputErrors(): Array<DenseNDArray> = arrayOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.2)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, 0.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.9))
  )

  /**
   *
   */
  fun buildNetwork(): NeuralNetwork {

    val network = NeuralNetwork(
      LayerConfiguration(
        size = 3,
        type = LayerType.Input.Dense),
      LayerConfiguration(
        size = 2,
        activationFunction = Tanh(),
        connectionType = LayerType.Connection.Feedforward
      ))

    initParameters(network.model.paramsPerLayer[0] as FeedforwardLayerParameters)

    return network
  }


  /**
   *
   */
  private fun initParameters(params: FeedforwardLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(-0.7, 0.3, -1.0),
      doubleArrayOf(0.8, -0.6, 0.4)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.9)))
  }
}
