/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.treernn

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.treernn.TreeRNN
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
}
