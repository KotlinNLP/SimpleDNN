/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayerParameters

class NormalizationLayerParameters (
    inputSize: Int,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer(),
    private val sparseInput: Boolean = false
) : LayerParameters<NormalizationLayerParameters>(
    inputSize = inputSize,
    outputSize = inputSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The bias array.
   */
  val b = ParamsArray(this.outputSize)

  /**
   * The bias array.
   */
  val g = ParamsArray(this.outputSize)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(
      this.g, this.b
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(
     this.g
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
      this.b
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [HighwayLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): NormalizationLayerParameters {

    val clonedParams = NormalizationLayerParameters(
        inputSize = this.inputSize,
        sparseInput = this.sparseInput,
        weightsInitializer = null,
        biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}