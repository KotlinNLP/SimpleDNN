/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableSparseArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The parameters of a layer
 *
 * @property inputSize input size
 * @property outputSize output size
 */
abstract class LayerParameters<ParamsType: LayerParameters<ParamsType>>(
  val inputSize: Int,
  val outputSize: Int
) : IterableParams<ParamsType>() {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   * @param randomGenerator randomGenerator
   * @param biasesInitValue biasesInitValue
   * @return
   */
  abstract fun initialize(
    randomGenerator: RandomGenerator = FixedRangeRandom(radius = 0.08, enablePseudoRandom = true),
    biasesInitValue: Double = 0.01)

  /**
   *
   */
  protected fun buildUpdatableArray(dim1: Int, dim2: Int = 1, sparseInput: Boolean = false): UpdatableArray<*> =
    if (sparseInput)
      this.buildSparseArray(dim1, dim2)
    else
      this.buildDenseArray(dim1, dim2)

  /**
   *
   */
  protected fun buildDenseArray(dim1: Int, dim2: Int = 1) = UpdatableDenseArray(Shape(dim1, dim2))

  /**
   *
   */
  private fun buildSparseArray(dim1: Int, dim2: Int = 1) = UpdatableSparseArray(Shape(dim1, dim2))
}
