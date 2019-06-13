/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters

class TPRLayerParameters(
    inputSize: Int,
    outputSize: Int,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer(),
    private val sparseInput: Boolean = false
) : LayerParameters<TPRLayerParameters>(
inputSize = inputSize,
outputSize = outputSize,
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
  override val paramsList: List<ParamsArray> = emptyList()

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = emptyList()

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = emptyList()

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [TPRLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): TPRLayerParameters {

    val clonedParams = TPRLayerParameters(
        inputSize = this.inputSize,
        outputSize = this.outputSize,
        sparseInput = this.sparseInput,
        weightsInitializer = null,
        biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }

}
