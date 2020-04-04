/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention.scaleddot

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ScaledDotAttentionLayerUtils {

  /**
   *
   */
  fun buildAttentionParams(initializer: Initializer? = null) =
    ScaledDotAttentionLayerParameters(
      inputSize = 4,
      attentionSize = 2,
      outputSize = 3,
      weightsInitializer = initializer
    ).apply {
      queries.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.3, 0.4, 0.2, -0.2),
        doubleArrayOf(0.2, -0.1, 0.1, 0.6)
      )))
      keys.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(-0.2, 0.8, 0.6, 0.8),
        doubleArrayOf(0.2, 0.8, 0.0, -0.6)
      )))
      values.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.9, 0.8, 0.5, 0.3),
        doubleArrayOf(-0.6, -0.2, 0.4, 0.4),
        doubleArrayOf(-0.6, -0.7, -0.4, 0.6)
      )))
    }

  /**
   *
   */
  fun buildInputSequence(): List<AugmentedArray<DenseNDArray>> = listOf(
    AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.7, 0.9, 0.6))),
    AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.7, -0.7, 0.8))),
    AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.5, 0.0, 0.2)))
  )

  /**
   *
   */
  fun buildOutputErrors(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.7)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.5, -0.5)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, -0.5, 0.2))
  )
}
