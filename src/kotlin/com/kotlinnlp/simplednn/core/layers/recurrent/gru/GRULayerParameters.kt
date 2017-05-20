/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.layers.recurrent.GateParametersUnit
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

/**
 *
 * @param inputSize input size
 * @param outputSize output size
 */
class GRULayerParameters(inputSize: Int, outputSize: Int) : LayerParameters(inputSize, outputSize) {

  /**
   *
   */
  val candidate = GateParametersUnit(inputSize, outputSize)

  /**
   *
   */
  val resetGate = GateParametersUnit(inputSize, outputSize)

  /**
   *
   */
  val partitionGate = GateParametersUnit(inputSize, outputSize)

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
