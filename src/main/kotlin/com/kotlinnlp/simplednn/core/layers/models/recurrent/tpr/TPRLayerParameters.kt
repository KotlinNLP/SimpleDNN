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
    val nSymbols: Int,
    val dSymbols: Int,
    val nRoles: Int,
    val dRoles: Int,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer(),
    private val sparseInput: Boolean = false
) : LayerParameters<TPRLayerParameters>(
    inputSize = inputSize,
    outputSize = dSymbols * dRoles,
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
   * The weights connecting input to the Symbol attention vector
   */
  val wInS = ParamsArray(this.nSymbols, this.inputSize)

  /**
   * The weights connecting input to the Role attention vector
   */
  val wInR = ParamsArray(this.nRoles, this.inputSize)

  /**
   * The weights connecting previous output to the Symbol attention vector
   */
  val wRecS = ParamsArray(this.nSymbols, this.inputSize)

  /**
   * The weights connecting previous output to the Role attention vector
   */
  val wRecR = ParamsArray(this.nRoles, this.inputSize)

  /**
   * The Symbol attention vector bias.
   */
  val bS = ParamsArray(this.nSymbols)

  /**
   * The Role attention vector bias.
   */
  val bR = ParamsArray(this.nRoles)

  /**
   * The Symbol attention embeddings.
   */
  val S = ParamsArray(this.dSymbols, this.nSymbols)

  /**
   * The Role attention embeddings.
   */
  val R = ParamsArray(this.dRoles, this.nRoles)

  /**
   * The list of all parameters.
   */
  override val paramsList: List<ParamsArray> = listOf(
      this.wRecS,
      this.wRecR,
      this.wInS,
      this.wInR,
      this.S,
      this.R,
      this.bS,
      this.bR
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(
      this.wRecS,
      this.wRecR,
      this.wInS,
      this.wInR,
      this.S,
      this.R
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
      this.bS,
      this.bR
  )

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
        nSymbols = this.nSymbols,
        dSymbols = this.dSymbols,
        nRoles = this.nRoles,
        dRoles = this.dRoles,
        sparseInput = this.sparseInput,
        weightsInitializer = null,
        biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
