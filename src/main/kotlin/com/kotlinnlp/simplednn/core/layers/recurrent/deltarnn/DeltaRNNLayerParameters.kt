/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.ParametersUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The parameters of the layer of type SimpleRecurrent.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class DeltaRNNLayerParameters(
  inputSize: Int,
  outputSize: Int,
  private val sparseInput: Boolean = false
) : LayerParameters<DeltaRNNLayerParameters>(inputSize = inputSize, outputSize = outputSize) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  val feedforwardUnit = ParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val recurrentUnit = ParametersUnit(inputSize = this.outputSize, outputSize = this.outputSize)

  /**
   *
   */
  val alpha = UpdatableDenseArray(shape = Shape(this.outputSize))

  /**
   *
   */
  val beta1 = UpdatableDenseArray(shape = Shape(this.outputSize))

  /**
   *
   */
  val beta2 = UpdatableDenseArray(shape = Shape(this.outputSize))

  /**
   * The list of all parameters.
   */
  override val paramsList = arrayOf(
    this.feedforwardUnit.weights,
    this.feedforwardUnit.biases,
    this.recurrentUnit.weights,
    this.recurrentUnit.biases,
    this.alpha,
    this.beta1,
    this.beta2
  )

  /**
   * Initialize values randomly.
   *
   * @param randomGenerator randomGenerator
   * @param biasesInitValue biasesInitValue
   */
  override fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double): DeltaRNNLayerParameters {
    require(!this.sparseInput) { "Cannot randomize sparse weights" }

    this.feedforwardUnit.weights.values.randomize(randomGenerator)
    this.feedforwardUnit.biases.values.assignValues(biasesInitValue)

    this.recurrentUnit.weights.values.randomize(randomGenerator)
    this.recurrentUnit.biases.values.assignValues(biasesInitValue)

    this.alpha.values.randomize(randomGenerator)
    this.beta1.values.randomize(randomGenerator)
    this.beta2.values.randomize(randomGenerator)

    return this
  }

  /**
   * @return a new [DeltaRNNLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): DeltaRNNLayerParameters {

    val clonedParams = DeltaRNNLayerParameters(
      inputSize = this.inputSize,
      outputSize = this.outputSize,
      sparseInput = this.sparseInput)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
