/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The Max Pooling Layer Parameters.
 *
 *  @param weightsInitializer the initializer of the weights (kernels)
 *  @param biasesInitializer the initializer of the bias
 */
class MaxPoolingLayerParameters (
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer(),
    private val sparseInput: Boolean = false
) : LayerParameters<MaxPoolingLayerParameters>(
    inputSize = 0,
    outputSize = 0,
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
   * The list of all parameters.
   */
  override val paramsList = emptyList<ParamsArray>()

  /**
   * The list of weights parameters.
   */
  override val weightsList = emptyList<ParamsArray>()

  /**
   * The list of biases parameters.
   */
  override val biasesList = emptyList<ParamsArray>()

  /**
   * @return a new [MaxPoolingLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): MaxPoolingLayerParameters {

    val clonedParams = MaxPoolingLayerParameters(
        sparseInput = this.sparseInput,
        weightsInitializer = null,
        biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}