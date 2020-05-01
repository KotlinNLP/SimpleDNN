/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.norm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object NormLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): NormLayer<DenseNDArray> = NormLayer(
    inputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.8, -0.7, -0.5))),
    outputArray = AugmentedArray.zeros(size = 4),
    inputType = LayerType.Input.Dense,
    params = buildParams())

  /**
   *
   */
  fun buildParams() = NormLayerParameters(inputSize = 4).apply {
    g.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8)))
    b.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2)))
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.6))
}
