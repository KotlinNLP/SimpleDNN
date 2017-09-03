/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers.affine

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.deeplearning.mergelayers.MergeLayerParameters

/**
 * The parameters of the affine layer.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
open class AffineLayerParameters(
  inputSize1: Int,
  inputSize2: Int,
  outputSize: Int,
  sparseInput: Boolean = false
) : MergeLayerParameters(
  inputSize1 = inputSize1,
  inputSize2 = inputSize2,
  outputSize = outputSize,
  sparseInput = sparseInput
) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The weights connected to the first input array.
   */
  val w1: UpdatableArray<*> = this.buildUpdatableArray(this.outputSize, this.inputSize1, sparseInput = this.sparseInput)

  /**
   * The parameters connected to the second input array.
   */
  val w2: UpdatableArray<*> = this.buildUpdatableArray(this.outputSize, this.inputSize2, sparseInput = this.sparseInput)

  /**
   * The bias array.
   */
  val b: UpdatableDenseArray = this.buildDenseArray(this.outputSize)

  /**
   * Initialize the parameters list.
   */
  init {
    this.paramsList = arrayListOf(
      this.w1,
      this.w2,
      this.b
    )
  }

  /**
   * Initialize all parameters with random or predefined values.
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.w1.values.randomize(randomGenerator)
    this.w2.values.randomize(randomGenerator)
    this.b.values.assignValues(biasesInitValue)
  }
}
