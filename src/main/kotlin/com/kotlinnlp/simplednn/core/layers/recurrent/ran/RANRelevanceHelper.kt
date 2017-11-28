/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.RelevanceUtils
import com.kotlinnlp.simplednn.core.layers.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [RANLayerStructure] in which to calculate the input relevance
 */
class RANRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayerStructure<InputNDArrayType>
) : GatedRecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Propagate the relevance from the output to the gate units of the layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun propagateRelevanceToGates(layerContributions: LayerParameters<*>) {

    val (inputRelevance, recurrentRelevance) = this.getRelevancePartitions(layerContributions as RANLayerParameters)
    val halfInputRelevance: DenseNDArray = inputRelevance.assignDiv(2.0)

    this.layer.candidate.assignRelevance(halfInputRelevance)
    this.layer.inputGate.assignRelevance(halfInputRelevance)

    if (recurrentRelevance != null) {
      this.layer.forgetGate.assignRelevance(recurrentRelevance.assignDiv(2.0))
    }
  }

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input in respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    layerContributions as RANLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val prevStateExists: Boolean = this.layer.layerContextWindow.getPrevStateLayer() != null

    val inputGateRelevance: NDArray<*> = this.layer.inputGate.getInputRelevance(
      x = x,
      contributions = layerContributions.inputGate,
      prevStateExists = prevStateExists)

    val candidateRelevance: NDArray<*> = this.layer.candidate.getInputRelevance(
      x = x,
      contributions = layerContributions.candidate)

    val inputRelevance: NDArray<*> = inputGateRelevance.assignSum(candidateRelevance)

    if (prevStateExists) {

      val forgetGateRelevance: NDArray<*> = this.layer.forgetGate.getInputRelevance(
        x = x,
        contributions = layerContributions.forgetGate,
        prevStateExists = true)

      inputRelevance.assignSum(forgetGateRelevance)
    }

    return inputRelevance
  }

  /**
   * Calculate the relevance of the output in the previous state respect of the current one and assign it to the output
   * array of the previous state.
   * WARNING: it's needed that a previous state exists.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(layerContributions: LayerParameters<*>) {
    layerContributions as RANLayerParameters

    val prevStateOutput = this.layer.layerContextWindow.getPrevStateLayer()!!.outputArray
    val (_, recurrentRelevance) = this.getRelevancePartitions(layerContributions)
    val halfRecurrentRelevance: DenseNDArray = recurrentRelevance!!.assignDiv(2.0)

    val inputGateRelevance: NDArray<*> = this.layer.inputGate.getRecurrentRelevance(
      contributions = layerContributions.inputGate,
      yPrev = prevStateOutput.values)

    val forgetGateRelevance: NDArray<*> = this.layer.forgetGate.getRecurrentRelevance(
      contributions = layerContributions.forgetGate,
      yPrev = prevStateOutput.values)

    prevStateOutput.assignRelevance(halfRecurrentRelevance)
    prevStateOutput.relevance.assignSum(inputGateRelevance).assignSum(forgetGateRelevance)
  }

  /**
   * Get the partitions of relevance respectively among the candidate and the previous state.
   * If there isn't a previous state its partition is null and all the output relevance is assigned to the candidate
   * partition.
   *
   * @param contributions the contributions saved during the last forward
   *
   * @return a pair containing the partitions of the output relevance
   */
  private fun getRelevancePartitions(contributions: RANLayerParameters): Pair<DenseNDArray, DenseNDArray?> {

    val yRelevance: DenseNDArray = this.layer.outputArray.relevance as DenseNDArray

    return if (this.layer.layerContextWindow.getPrevStateLayer() != null) {
      this.splitRelevancePartitions(yRelevance = yRelevance, contributions = contributions)

    } else {
      Pair(yRelevance, null)
    }
  }

  /**
   * @param yRelevance the relevance of the output array
   * @param contributions the contributions saved during the last forward
   *
   * @return a pair containing the partitions of relevance respectively among the candidate and the previous state
   */
  private fun splitRelevancePartitions(yRelevance: DenseNDArray,
                                       contributions: RANLayerParameters): Pair<DenseNDArray, DenseNDArray?> {

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = contributions.candidate.biases.values as DenseNDArray
    val yInput: DenseNDArray = y.sub(yRec)

    val inputRelevance: DenseNDArray = RelevanceUtils.getRelevancePartition1(
      yRelevance = yRelevance,
      y = y,
      yContribute1 = yInput,
      yContribute2 = yRec)

    val recurrentRelevance: DenseNDArray = RelevanceUtils.getRelevancePartition2(
      yRelevance = yRelevance,
      y = y,
      yContribute2 = yRec)

    return Pair(inputRelevance, recurrentRelevance)
  }
}
