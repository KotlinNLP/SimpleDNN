/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which return the structures involved in roles and symbols scores calculation.
 */
class TPRScoresHelper {

  /**
   * Calculate the euclidean norm of the vector v
   */
  private fun norm(v: DenseNDArray): Double{
    var norm = 0.0
    for (i in 0 until v.length)
      norm += v[i] * v[i]

    return norm
  }

  /**
   * Get the normalized role vector.
   */
  fun getRolesScores(layer: TPRLayer<DenseNDArray>): DenseNDArray {

    val roleVector = layer.aR.values.copy()

    for (i in 0 until roleVector.length)
      roleVector[i] = roleVector[i] / norm(roleVector)

    return roleVector
  }

  /**
   * Get the symbol vector.
   */
  fun getSymbolEmbedding(layer: TPRLayer<DenseNDArray>): DenseNDArray {

    return layer.s.values.copy()
  }

  /**
   * Get the TPR layer.
   */
  fun getTprLayer(layers: List<Layer<*>>): TPRLayer<DenseNDArray> {
    @Suppress("UNCHECKED_CAST")
    return layers[0] as TPRLayer<DenseNDArray>
  }
}