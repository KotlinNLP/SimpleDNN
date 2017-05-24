/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.layers.recurrent.GateParametersUnit
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters

/**
 *
 * @param inputSize input size
 * @param outputSize output size
 */
class RANLayerParameters(inputSize: Int, outputSize: Int) : LayerParameters(inputSize, outputSize) {

  /**
   *
   */
  val inputGate = GateParametersUnit(inputSize, outputSize)

  /**
   *
   */
  val forgetGate = GateParametersUnit(inputSize, outputSize)

  /**
   *
   */
  val candidate = FeedforwardLayerParameters(inputSize, outputSize)

  /**
   *
   */
  init {

    this.paramsList = arrayListOf(
      this.inputGate.weights,
      this.forgetGate.weights,
      this.candidate.weights,

      this.inputGate.biases,
      this.forgetGate.biases,
      this.candidate.biases,

      this.inputGate.recurrentWeights,
      this.forgetGate.recurrentWeights
    )
  }

  /**
   *
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {

    this.inputGate.weights.values.randomize(randomGenerator)
    this.forgetGate.weights.values.randomize(randomGenerator)
    this.candidate.weights.values.randomize(randomGenerator)

    this.inputGate.biases.values.assignValues(biasesInitValue)
    this.forgetGate.biases.values.assignValues(biasesInitValue)
    this.candidate.biases.values.assignValues(biasesInitValue)

    this.inputGate.recurrentWeights.values.randomize(randomGenerator)
    this.forgetGate.recurrentWeights.values.randomize(randomGenerator)
  }
}
