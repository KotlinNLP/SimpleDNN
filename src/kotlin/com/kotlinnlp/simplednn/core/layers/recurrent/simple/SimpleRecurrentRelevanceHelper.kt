/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.core.layers.RelevanceUtils
import com.kotlinnlp.simplednn.core.layers.recurrent.RecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

typealias RU = RelevanceUtils

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which to calculate the input relevance
 */
class SimpleRecurrentRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : RecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters): NDArray<*> {
    layerContributions as SimpleRecurrentLayerParameters

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = layerContributions.biases.values
    val yInput: DenseNDArray = y.sub(yRec)
    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray
    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? SimpleRecurrentLayerStructure<*>

    return RelevanceUtils.calculateRelevanceOfArray(
      x = this.layer.inputArray.values,
      y = yInput,
      yRelevance = if (prevStateLayer != null)
        RU.getRelevancePartition1(yRelevance = yRelevance, y = y, yContribute1 = yInput, yContribute2 = yRec)
      else
        this.layer.outputArray.relevance as DenseNDArray,
      contributions = layerContributions.weights.values
    )
  }

  /**
   * Calculate the relevance of the output in the previous state respect of the current one and assign it to the output
   * array of the previous state.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(layerContributions: LayerParameters) {
    layerContributions as SimpleRecurrentLayerParameters

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = layerContributions.biases.values
    val prevStateLayer: LayerStructure<*> = this.layer.layerContextWindow.getPrevStateLayer()!!
    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray

    val recurrentRelevance = RU.calculateRelevanceOfDenseArray(
      x = prevStateLayer.outputArray.values,
      y = yRec,
      yRelevance = RU.getRelevancePartition2(yRelevance = yRelevance, y = y, yContribute2 = yRec),
      contributions = layerContributions.recurrentWeights.values
    )

    prevStateLayer.outputArray.assignRelevance(recurrentRelevance)
  }
}
