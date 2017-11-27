/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork

import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * [NetworkParameters] contains all the parameters of the layers defined in [layersConfiguration],
 * grouped per layer.
 *
 * @property layersConfiguration a list of configurations, one per layer
 * @property sparseInput whether the input is sparse or not
 */
class NetworkParameters(
  val layersConfiguration: List<LayerConfiguration>,
  private val sparseInput: Boolean = false
) : IterableParams<NetworkParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * An [Array] containing a [LayerParameters] for each layer.
   *
   * In [layersConfiguration] layers are defined as a list [x, y, z], but the structure
   * contains layers as input-output pairs [x-y, y-z].
   * The output of a layer is a reference of the input of the next layer.
   */
  val paramsPerLayer: Array<LayerParameters<*>> = Array(
    size = layersConfiguration.size - 1,
    init = { i ->
      LayerParametersFactory(
        inputSize = layersConfiguration[i].size,
        outputSize = layersConfiguration[i + 1].size,
        connectionType = layersConfiguration[i + 1].connectionType!!,
        sparseInput = this.sparseInput && i == 0)
    }
  )

  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> = this.buildParamsList()

  /**
   * @return a new [NetworkParameters] containing a copy of all parameters of this
   */
  override fun copy(): NetworkParameters {

    val clonedParams = NetworkParameters(layersConfiguration = this.layersConfiguration, sparseInput = this.sparseInput)

    clonedParams.zip(this) { cloned, params ->
      cloned.values.assignValues(params.values)
    }

    return clonedParams
  }

  /**
   * @return the list with parameters of all layers
   */
  private fun buildParamsList(): Array<UpdatableArray<*>> {

    var layerIndex = 0
    var paramIndex = 0

    return Array(
      size = this.paramsPerLayer.sumBy { it.size },
      init = {

        if (paramIndex == this.paramsPerLayer[layerIndex].size) {
          layerIndex++
          paramIndex = 0
        }

        this.paramsPerLayer[layerIndex][paramIndex++] }
    )
  }
}
