/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.layers.RecurrentParametersUnit
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

/**
 * The parameters of the layer of type LSTM.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class LSTMLayerParameters(
  inputSize: Int,
  outputSize: Int,
  private val sparseInput: Boolean = false
) : LayerParameters(inputSize = inputSize, outputSize = outputSize) {

  /**
   *
   */
  val inputGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val outputGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val forgetGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

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
  init {

    this.paramsList = arrayListOf(
      this.inputGate.weights,
      this.outputGate.weights,
      this.forgetGate.weights,
      this.candidate.weights,

      this.inputGate.biases,
      this.outputGate.biases,
      this.forgetGate.biases,
      this.candidate.biases,

      this.inputGate.recurrentWeights,
      this.outputGate.recurrentWeights,
      this.forgetGate.recurrentWeights,
      this.candidate.recurrentWeights
    )
  }

  /**
   *
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.inputGate.weights.values.randomize(randomGenerator)
    this.outputGate.weights.values.randomize(randomGenerator)
    this.forgetGate.weights.values.randomize(randomGenerator)
    this.candidate.weights.values.randomize(randomGenerator)

    this.inputGate.biases.values.assignValues(biasesInitValue)
    this.outputGate.biases.values.assignValues(biasesInitValue)
    this.forgetGate.biases.values.assignValues(biasesInitValue)
    this.candidate.biases.values.assignValues(biasesInitValue)

    this.inputGate.recurrentWeights.values.randomize(randomGenerator)
    this.outputGate.recurrentWeights.values.randomize(randomGenerator)
    this.forgetGate.recurrentWeights.values.randomize(randomGenerator)
    this.candidate.recurrentWeights.values.randomize(randomGenerator)
  }
}
