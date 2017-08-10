/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.layers.RecurrentParametersUnit
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

/**
 * The parameters of the layer of type GRU.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class GRULayerParameters(
  inputSize: Int,
  outputSize: Int,
  private val sparseInput: Boolean = false
) : LayerParameters(inputSize = inputSize, outputSize = outputSize) {

  /**
   *
   */
  val candidate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val resetGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val partitionGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  init {

    this.paramsList = arrayListOf(
      this.candidate.weights,
      this.resetGate.weights,
      this.partitionGate.weights,

      this.candidate.biases,
      this.resetGate.biases,
      this.partitionGate.biases,

      this.candidate.recurrentWeights,
      this.resetGate.recurrentWeights,
      this.partitionGate.recurrentWeights
    )
  }

  /**
   *
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.candidate.weights.values.randomize(randomGenerator)
    this.resetGate.weights.values.randomize(randomGenerator)
    this.partitionGate.weights.values.randomize(randomGenerator)

    this.candidate.biases.values.assignValues(biasesInitValue)
    this.resetGate.biases.values.assignValues(biasesInitValue)
    this.partitionGate.biases.values.assignValues(biasesInitValue)

    this.candidate.recurrentWeights.values.randomize(randomGenerator)
    this.resetGate.recurrentWeights.values.randomize(randomGenerator)
    this.partitionGate.recurrentWeights.values.randomize(randomGenerator)
  }
}
