/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * [StackedLayersParameters] contains all the parameters of the layers defined in [layersConfiguration],
 * grouped per layer.
 *
 * @property layersConfiguration a list of configurations, one per layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @param forceDense force all parameters to be dense (false by default)
 */
class StackedLayersParameters(
  val layersConfiguration: List<LayerInterface>,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  private val forceDense: Boolean = false
) : IterableParams<StackedLayersParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * A list containing a [LayerParameters] for each layer.
   */
  val paramsPerLayer: List<LayerParameters<*>> = this.layersConfiguration.toLayerParameters(
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer,
    forceDenseInput = this.forceDense
  )

  /**
   * The list of all parameters.
   */
  override val paramsList: List<UpdatableArray<*>> = this.paramsPerLayer.toUpdatableArrays()

  /**
   * @return a new [StackedLayersParameters] containing a copy of all parameters of this
   */
  override fun copy(): StackedLayersParameters {

    val clonedParams = StackedLayersParameters(
      layersConfiguration = this.layersConfiguration,
      forceDense = this.forceDense,
      weightsInitializer = null,
      biasesInitializer = null)

    clonedParams.zip(this) { cloned, params ->
      cloned.values.assignValues(params.values)
    }

    return clonedParams
  }
}
