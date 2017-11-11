/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.RecurrentParametersUnit

/**
 * The parameters of the layer of type SimpleRecurrent.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class SimpleRecurrentLayerParameters(
  inputSize: Int,
  outputSize: Int,
  private val sparseInput: Boolean = false
) : LayerParameters<SimpleRecurrentLayerParameters>(inputSize = inputSize, outputSize = outputSize) {

  /**
   *
   */
  val unit = RecurrentParametersUnit(
    outputSize = this.outputSize,
    inputSize = this.inputSize,
    sparseInput = this.sparseInput)

  /**
   * The list of all parameters.
   */
  override val paramsList = arrayOf(
    this.unit.weights,
    this.unit.biases,
    this.unit.recurrentWeights
  )

  /**
   *
   * @param randomGenerator randomGenerator
   * @param biasesInitValue biasesInitValue
   * @return
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double): SimpleRecurrentLayerParameters {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.unit.weights.values.randomize(randomGenerator)
    this.unit.biases.values.assignValues(biasesInitValue)
    this.unit.recurrentWeights.values.randomize(randomGenerator)

    return this
  }

  /**
   * @return a new [SimpleRecurrentLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): SimpleRecurrentLayerParameters {

    val clonedParams = SimpleRecurrentLayerParameters(
      inputSize = this.inputSize,
      outputSize = this.outputSize,
      sparseInput = this.sparseInput)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
