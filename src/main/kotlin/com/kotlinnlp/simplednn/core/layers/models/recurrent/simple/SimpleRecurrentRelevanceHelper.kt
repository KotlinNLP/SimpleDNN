/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [SimpleRecurrentLayer] in which to calculate the input relevance
 */
class SimpleRecurrentRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayer<InputNDArrayType>
) : RecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    layerContributions as SimpleRecurrentLayerParameters

    return this.layer.outputArray.getInputRelevance(
      x = this.layer.inputArray.values,
      contributions = layerContributions.unit,
      prevStateExists = this.layer.layerContextWindow.getPrevState() != null)
  }

  /**
   * Calculate the relevance of the output in the previous state in respect of the current one and assign it to the
   * output array of the previous state.
   * WARNING: it's needed that a previous state exists.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(layerContributions: LayerParameters<*>) {
    layerContributions as SimpleRecurrentLayerParameters

    val prevStateLayer: Layer<*> = this.layer.layerContextWindow.getPrevState()!!
    val recurrentRelevance: DenseNDArray = this.layer.outputArray.getRecurrentRelevance(
      contributions = layerContributions.unit,
      yPrev = prevStateLayer.outputArray.values)

    prevStateLayer.outputArray.assignRelevance(recurrentRelevance)
  }
}
