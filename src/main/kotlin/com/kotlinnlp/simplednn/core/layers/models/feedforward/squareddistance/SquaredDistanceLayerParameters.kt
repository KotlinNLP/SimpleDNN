/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of the layer of type Squared Distance.
 * See Hewitt, Manning, A structural probe for finding Syntax in word representation paragraph 2.1
 * The matrix B  k x m where m is [inputSize]. k is [outputSize], the size of the transformed vector whose norm is the
 * layer output.
 *
 * @property inputSize input size
 * @property outputSize transformed vector size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 */
class SquaredDistanceLayerParameters(
    inputSize: Int,
    outputSize: Int,
    weightsInitializer: Initializer? = GlorotInitializer()
) : LayerParameters<SquaredDistanceLayerParameters>(
    inputSize = inputSize,
    outputSize = outputSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = null) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The weights connected to the first input array.
   */
  val B = ParamsArray(this.outputSize, this.inputSize)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(this.B)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(this.B)

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
   * @return a new [SquaredDistanceLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): SquaredDistanceLayerParameters {

    val clonedParams = SquaredDistanceLayerParameters(
        outputSize = this.outputSize,
        inputSize = this.inputSize,
        weightsInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
