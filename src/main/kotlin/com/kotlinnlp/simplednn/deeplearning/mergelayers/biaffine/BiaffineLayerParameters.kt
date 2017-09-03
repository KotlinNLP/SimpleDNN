/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers.biaffine

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.deeplearning.mergelayers.affine.AffineLayerParameters

/**
 * The parameters of the biaffine layer.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class BiaffineLayerParameters(
  inputSize1: Int,
  inputSize2: Int,
  outputSize: Int,
  sparseInput: Boolean = false
) : AffineLayerParameters(
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
   * The weights connected to each the first and the second input arrays.
   */
  val w: Array<UpdatableArray<*>> = Array(
    size = this.outputSize,
    init = { this.buildUpdatableArray(this.inputSize2, this.inputSize1, sparseInput = this.sparseInput) }
  )

  /**
   * Initialize the parameters list.
   */
  init {

    this.paramsList = arrayListOf(
      this.w1,
      this.w2,
      this.b
    )

    this.w.forEach { this.paramsList.add(it) }
  }

  /**
   * Initialize all parameters with random or predefined values.
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.w1.values.randomize(randomGenerator)
    this.w2.values.randomize(randomGenerator)
    this.b.values.assignValues(biasesInitValue)

    this.w.forEach { it.values.randomize(randomGenerator) }
  }
}
