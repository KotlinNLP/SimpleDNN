/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.RecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which to calculate the input relevance
 */
class SimpleRecurrentRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : RecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Calculate the relevance of the input respect of the output.
   *
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters): NDArray<*> {
    layerContributions as SimpleRecurrentLayerParameters

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = layerContributions.biases.values
    val yInput: DenseNDArray = y.sub(yRec)
    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? SimpleRecurrentLayerStructure<*>

    return this.calculateRelevanceOfArray(
      x = this.layer.inputArray.values,
      y = yInput,
      yRelevance = if (prevStateLayer != null)
        this.getInputRelevancePartition(y = y, yInput = yInput, yRec = yRec)
      else
        this.layer.outputArray.relevance as DenseNDArray,
      contributions = layerContributions.weights.values
    )
  }

  /**
   * Calculate the relevance of the output in the previous state respect of the current one.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun calculateRecurrentRelevance(layerContributions: LayerParameters) {
    layerContributions as SimpleRecurrentLayerParameters

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = layerContributions.biases.values
    val prevStateLayer: LayerStructure<*> = this.layer.layerContextWindow.getPrevStateLayer()!!

    val recurrentRelevance = this.calculateRelevanceOfDenseArray(
      x = prevStateLayer.outputArray.values,
      y = yRec,
      yRelevance = this.getRecurrentRelevancePartition(y = y, yRec = yRec),
      contributions = layerContributions.recurrentWeights.values
    )

    prevStateLayer.outputArray.assignRelevance(recurrentRelevance)
  }

  /**
   * Get the partition of the output relevance respect of the input.
   *
   * @param y the output array of the layer
   * @param yInput the contribution of the input to calculate the output array
   * @param yRec the contribution of the recursion to calculate the output array
   *
   * @return the partition of the output relevance respect of the input
   */
  private fun getInputRelevancePartition(y: DenseNDArray, yInput: DenseNDArray, yRec: DenseNDArray): DenseNDArray {

    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray
    val eps: DenseNDArray = yRec.nonZeroSign().assignProd(this.relevanceEps) // the same factor (yRec) is needed
    // to calculate eps either for the input partition then the recurrent one

    // partition factor = (yInput + eps / 2) / (yInput + yRec + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yInput.sum(eps.div(2.0))).assignDiv(y.sum(eps))
  }

  /**
   * Get the partition of the output relevance respect of the output in the previous state.
   *
   * @param y the output array of the layer
   * @param yRec the contribution of the recursion to calculate the output array
   *
   * @return the partition of the output relevance respect of the output in the previous state
   */
  private fun getRecurrentRelevancePartition(y: DenseNDArray, yRec: DenseNDArray): DenseNDArray {

    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray
    val eps: DenseNDArray = yRec.nonZeroSign().assignProd(this.relevanceEps)

    // partition factor = (yRec + eps / 2) / (yInput + yRec + eps) [eps avoids divisions by zero]
    return yRelevance.prod(yRec.sum(eps.div(2.0))).assignDiv(y.sum(eps))
  }
}
