/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package optimizer

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory

/**
 *
 */
object IterableParamsUtils {

  /**
   *
   */
  fun buildDenseParams1(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 3, outputSize = 2)

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.4, 0.8, 0.2),
      doubleArrayOf(0.1, 0.3, 0.9)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1)))

    return params
  }

  /**
   *
   */
  fun buildDenseParams2(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 3, outputSize = 2)

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.1, -0.5, 0.3),
      doubleArrayOf(-0.9, 0.0, 0.6)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.1)))

    return params
  }

  /**
   *
   */
  fun buildSparseParams1(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 3, outputSize = 2, sparseInput = true)

    params.unit.weights.values.assignValues(SparseNDArrayFactory.arrayOf(
      activeIndicesValues = arrayOf(
        SparseEntry(Indices(0, 0), 0.4),
        SparseEntry(Indices(1, 1), 0.3)
      ),
      shape = Shape(2, 3)
    ))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1)))

    return params
  }

  /**
   *
   */
  fun buildSparseParams2(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 3, outputSize = 2, sparseInput = true)

    params.unit.weights.values.assignValues(SparseNDArrayFactory.arrayOf(
      activeIndicesValues = arrayOf(
        SparseEntry(Indices(1, 1), -0.5),
        SparseEntry(Indices(1, 2), 0.6)
      ),
      shape = Shape(2, 3)
    ))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.1)))

    return params
  }
}
