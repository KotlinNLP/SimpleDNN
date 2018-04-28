/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.treernn

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.encoders.treernn.TreeEncoder
import com.kotlinnlp.simplednn.encoders.treernn.TreeRNN
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object TreeRNNUtils {

  /**
   *
   */
  fun buildTreeRNN(): TreeRNN {

    val network = TreeRNN(
      inputLayerSize = 2,
      hiddenLayerSize = 3,
      hiddenLayerConnectionType = LayerType.Connection.SimpleRecurrent)

    val leftParams = network.leftRNN.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val rightParams = network.rightRNN.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val concatParams = network.concatNetwork.model.paramsPerLayer[0] as FeedforwardLayerParameters

    leftParams.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.6, 0.8),
      doubleArrayOf(-0.3, 0.0),
      doubleArrayOf(0.9, -0.8)
    )))

    leftParams.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.2, -0.3)))

    leftParams.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.1, 0.7, 0.0),
      doubleArrayOf(0.2, 0.9, -0.2),
      doubleArrayOf(-0.5, -0.2, -0.4)
    )))

    rightParams.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.1, 0.9),
      doubleArrayOf(-1.0, -0.4),
      doubleArrayOf(0.4, -0.8)
    )))

    rightParams.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.9, 0.4)))

    rightParams.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(1.0, -0.1, 0.7),
      doubleArrayOf(-0.7, 0.8, -1.0),
      doubleArrayOf(0.0, 0.8, 0.0)
    )))

    concatParams.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.2, -0.2, -0.4, 1.0, -0.5, -0.4),
      doubleArrayOf(0.5, 0.5, 0.2, -0.8, 0.5, 0.1)
    )))

    concatParams.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, -0.8)))

    return network
  }

  /**
   *
   */
  fun buildNodes(): Map<Int, DenseNDArray> = mapOf(
    Pair(1, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, -1.0))),
    Pair(2, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9))),
    Pair(3, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.9))),
    Pair(4, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.7))),
    Pair(5, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.2))),
    Pair(6, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, -0.5))),
    Pair(7, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.7))),
    Pair(8, DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.7)))
  )

  /**
   *
   */
  fun setHeads(treeEncoder: TreeEncoder) {

    treeEncoder.setHead(4, headId = 3)
    treeEncoder.setHead(5, headId = 3)
    treeEncoder.setHead(3, headId = 2)
    treeEncoder.setHead(1, headId = 2)
    treeEncoder.setHead(7, headId = 6)
    treeEncoder.setHead(8, headId = 7)
  }

  /**
   *
   */
  fun setHeads2(treeEncoder: TreeEncoder) {

    treeEncoder.setHead(3, headId = 2)
    treeEncoder.setHead(1, headId = 2)
    treeEncoder.setHead(4, headId = 3)
    treeEncoder.setHead(8, headId = 7)
    treeEncoder.setHead(7, headId = 6)
    treeEncoder.setHead(5, headId = 3)
  }

  /**
   *
   */
  fun getEncodingErrors(): Map<Int, DenseNDArray> = mapOf(
    Pair(2, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.4))),
    Pair(3, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, 0.3))),
    Pair(6, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.6))),
    Pair(8, DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.1)))
  )
}
