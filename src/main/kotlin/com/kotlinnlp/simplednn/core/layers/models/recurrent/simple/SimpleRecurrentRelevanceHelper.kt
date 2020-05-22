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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [SimpleRecurrentLayer] in which to calculate the input relevance
 */
internal class SimpleRecurrentRelevanceHelper(
  override val layer: SimpleRecurrentLayer<DenseNDArray>
) : RecurrentRelevanceHelper(layer) {

  /**
   * @param contributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect to the output
   */
  override fun getInputRelevance(contributions: LayerParameters): DenseNDArray =
    this.layer.outputArray.getInputRelevance(
      x = this.layer.inputArray.values,
      contributions = (contributions as SimpleRecurrentLayerParameters).unit,
      prevStateExists = this.layer.layersWindow.getPrevState() != null)

  /**
   * Calculate the relevance of the output in the previous state respect to the current one and assign it to the output
   * array of the previous state.
   *
   * WARNING: the previous state must exist!
   *
   * @param contributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(contributions: LayerParameters) {

    val prevStateLayer: Layer<*> = this.layer.layerContextWindow.getPrevState()!!
    val recurrentRelevance: DenseNDArray = this.layer.outputArray.getRecurrentRelevance(
      contributions = (contributions as SimpleRecurrentLayerParameters).unit,
      yPrev = prevStateLayer.outputArray.values)

    prevStateLayer.outputArray.assignRelevance(recurrentRelevance)
  }
}
