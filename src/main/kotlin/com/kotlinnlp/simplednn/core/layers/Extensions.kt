/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow

/**
 * Build a list containing a [LayerParameters] for each layer.
 *
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
fun List<LayerInterface>.toLayerParameters(
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
): List<LayerParameters<*>> = List(size = this.size - 1, init = { i ->
  LayerParametersFactory(
    inputsSize = this[i].sizes,
    outputSize = this[i + 1].size,
    connectionType = this[i + 1].connectionType!!,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer,
    sparseInput = this[i].type == LayerType.Input.SparseBinary
  )
})

/**
 * Build a new list of layer.
 *
 * Layers are defined as a list [x, y, z] of [LayerInterface].
 * The resulting list of [Layer] consist in input-output pairs [x-y, y-z].
 *
 * @param paramsPerLayer the parameters per layer
 * @param contextWindow the layer context window (can be null)
 *
 * @return list of layers where the output of a layer is the reference of the input of the next one
 */
fun List<LayerInterface>.toLayers(paramsPerLayer: List<LayerParameters<*>>,
                                  contextWindow: LayerContextWindow? = null): List<Layer<*>> {

  require(this.subList(1, this.size).all { it.type == LayerType.Input.Dense }) {
    "The last layers must be dense."
  }

  require(this.subList(2, this.size).all { it.connectionType!!.property != LayerType.Property.Merge }) {
    "Only the first layer can be a Merge layer."
  }

  var prevLayer: Layer<*>? = null

  return List(
    size = this.size - 1,
    init = { i ->

      val layer: Layer<*> = if (i == 0)
        LayerFactory(
          inputConfiguration = this[0],
          outputConfiguration = this[1],
          params = paramsPerLayer[0],
          contextWindow = contextWindow
        )
      else
        LayerFactory(
          inputArrays = listOf(prevLayer!!.outputArray),
          outputConfiguration = this[i + 1],
          params = paramsPerLayer[i],
          dropout = this[i].dropout,
          contextWindow = contextWindow
        )

      prevLayer = layer

      layer
    }
  )
}

/**
 * @return the list with parameters of all layers
 */
fun List<LayerParameters<*>>.toParamsArrays(): List<ParamsArray> {

  var layerIndex = 0
  var paramIndex = 0

  return List(
    size = this.sumBy { it.size },
    init = {

      if (paramIndex == this[layerIndex].size) {
        layerIndex++
        paramIndex = 0
      }

      this[layerIndex][paramIndex++]
    }
  )
}