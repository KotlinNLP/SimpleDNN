/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.cosinesimilarity

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity.CosineLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity.CosineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object CosineLayerUtils {

  /**
   *
   */
  fun buildLayer(): CosineLayer = CosineLayer(
      inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8, 0.6))),
      inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4, 0.8, -0.7))),
      params = CosineLayerParameters(inputSize = 4))

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.95))
}