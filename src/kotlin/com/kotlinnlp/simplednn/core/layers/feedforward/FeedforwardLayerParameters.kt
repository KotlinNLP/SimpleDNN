/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 * @param inputSize input size
 * @param outputSize output size
 */
class FeedforwardLayerParameters(inputSize: Int, outputSize: Int) : LayerParameters(inputSize, outputSize) {

  /**
   *
   */
  val weights: UpdatableArray = UpdatableArray(Shape(outputSize, inputSize))

  /**
   *
   */
  val biases: UpdatableArray = UpdatableArray(Shape(outputSize))

  /**
   *
   */
  init {
    this.paramsList = arrayListOf(
      this.weights,
      this.biases
    )
  }

  /**
   *
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double): Unit {
    this.weights.values.randomize(randomGenerator)
    this.biases.values.assignValues(biasesInitValue)
  }
}
