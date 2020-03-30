/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceUtils
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [DeltaRNNLayer] in which to calculate the input relevance
 */
internal class DeltaRNNRelevanceHelper(
  override val layer: DeltaRNNLayer<DenseNDArray>
) : GatedRecurrentRelevanceHelper(layer) {

  /**
   * Propagate the relevance from the output to the array units of the layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun propagateRelevanceToGates(layerContributions: LayerParameters) {
    layerContributions as DeltaRNNLayerParameters

    val previousStateExists: Boolean = this.layer.layerContextWindow.getPrevState() != null

    val halfOutputRelevance: DenseNDArray = this.layer.outputArray.relevance.div(2.0)

    val candidateRelevance: DenseNDArray = if (previousStateExists)
      this.getInputPartition(layerContributions).div(2.0)
    else
    // if there isn't a previous state, all the output relevance is assigned to the candidate
    // partition (p * c), half to the partition array and half to the candidate array
      halfOutputRelevance

    // pRelevance = inputPartition / 2 + recurrentPartition / 2 = outputRelevance / 2
    this.layer.partition.assignRelevance(halfOutputRelevance)
    this.layer.candidate.assignRelevance(candidateRelevance)

    this.setCandidateRelevancePartitions(previousStateExists = previousStateExists)
  }

  /**
   * @param layerContributions the contributions saved during the last forward
   *
   * @return the relevance of the input respect of the output
   */
  override fun getInputRelevance(layerContributions: LayerParameters): DenseNDArray {

    layerContributions as DeltaRNNLayerParameters

    val x = this.layer.inputArray.values

    val wxContrib: DenseNDArray = layerContributions.feedforwardUnit.weights.values

    val relevanceSupport: DeltaRNNRelevanceSupport = this.layer.relevanceSupport
    val previousStateExists: Boolean = this.layer.layerContextWindow.getPrevState() != null

    val bp: DenseNDArray = this.layer.params.recurrentUnit.biases.values
    val bc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val d1Bc: DenseNDArray = if (previousStateExists) bc.div(2.0) else bc
    // if there is a recurrent contribution bc is divided equally among d1Input and d1Rec, otherwise it is all assigned
    // to d1Input

    val pInputRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = this.layer.partition.valuesNotActivated,
      yRelevance = this.layer.partition.relevance,
      contributions = wxContrib.copy().partialAssignSum(bp) // w (dot) x + bp
    )

    val d1InputRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = relevanceSupport.d1Input.values,
      yRelevance = relevanceSupport.d1Input.relevance,
      contributions = wxContrib.prod(beta1).partialAssignSum(d1Bc) // (w (dot) x) * beta1 + (bc | bc / 2)
    )

    val inputRelevance: DenseNDArray = pInputRelevance.assignSum(d1InputRelevance)

    if (previousStateExists) {
      val d2InputRelevance = RelevanceUtils.calculateRelevanceOfArray(
        x = x,
        // the product by 'alpha' is not included during the calculation of the relevance
        // ('alpha' doesn't depend on variables of interest)
        y = this.layer.wx.values,
        yRelevance = relevanceSupport.d2.relevance.div(2.0),
        contributions = wxContrib // w (dot) x
      )

      inputRelevance.assignSum(d2InputRelevance)
    }

    return inputRelevance
  }

  /**
   * Calculate the relevance of the output in the previous state in respect of the current one and assign it to the
   * output array of the previous state.
   * WARNING: a previous state must exist.
   *
   * @param layerContributions the contributions saved during the last forward
   */
  override fun setRecurrentRelevance(layerContributions: LayerParameters) {

    layerContributions as DeltaRNNLayerParameters

    val prevStateOutput: AugmentedArray<DenseNDArray> = this.layer.layerContextWindow.getPrevState()!!.outputArray
    val yPrev: DenseNDArray = prevStateOutput.values

    val wyRecContrib: DenseNDArray = layerContributions.recurrentUnit.weights.values

    val halfBc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values.div(2.0)
    val beta2: DenseNDArray = this.layer.params.beta2.values

    val relevanceSupport: DeltaRNNRelevanceSupport = this.layer.relevanceSupport

    val d1RecRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = yPrev,
      y = relevanceSupport.d1Rec.values,
      yRelevance = relevanceSupport.d1Rec.relevance,
      contributions = wyRecContrib.prod(beta2).partialAssignSum(halfBc) // (wyRec (dot) yPrev) * beta2 + bc / 2
    )

    val d2RecRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = yPrev,
      // the product by 'alpha' is not included during the calculation of the relevance
      // ('alpha' doesn't depend on variables of interest)
      y = this.layer.wyRec.values,
      yRelevance = relevanceSupport.d2.relevance.div(2.0),
      contributions = wyRecContrib // wyRec (dot) yPrev
    )

    prevStateOutput.assignRelevance(this.getRecurrentPartition(layerContributions).div(2.0))
    prevStateOutput.relevance.assignSum(d1RecRelevance).assignSum(d2RecRelevance)
  }

  /**
   * Get the partition of relevance in respect of the input.
   * WARNING: a previous state must exist.
   *
   * @param contributions the contributions saved during the last forward
   *
   * @return the input relevance partition
   */
  private fun getInputPartition(contributions: DeltaRNNLayerParameters): DenseNDArray {

    val y: DenseNDArray = this.layer.outputArray.valuesNotActivated
    val yRec: DenseNDArray = contributions.recurrentUnit.biases.values
    val yInput: DenseNDArray = y.sub(yRec)

    return RelevanceUtils.getRelevancePartition1(
      yRelevance = this.layer.outputArray.relevance,
      y = y,
      yContribute1 = yInput,
      yContribute2 = yRec)
  }

  /**
   * Get the partition of relevance in respect of the previous state.
   *
   * @param contributions the contributions saved during the last forward
   *
   * @return the recurrent relevance partition
   */
  private fun getRecurrentPartition(contributions: DeltaRNNLayerParameters): DenseNDArray {

    return RelevanceUtils.getRelevancePartition2(
      yRelevance = this.layer.outputArray.relevance,
      y = this.layer.outputArray.valuesNotActivated,
      yContribute2 = contributions.recurrentUnit.biases.values)
  }

  /**
   * Set the partitions of the relevance of the candidate splitting it respectively among the d1 and d2 arrays.
   * The d1 array is itself composed by an input partition and a recurrent partition.
   * If there isn't a previous state all the candidate relevance is assigned to the d1 input partition.
   */
  private fun setCandidateRelevancePartitions(previousStateExists: Boolean) {

    val cRelevance: DenseNDArray = this.layer.candidate.relevance
    val relevanceSupport: DeltaRNNRelevanceSupport = this.layer.relevanceSupport

    if (previousStateExists) {
      this.splitCandidateRelevancePartitions(cRelevance = cRelevance, relevanceSupport = relevanceSupport)

    } else {
      relevanceSupport.d1Input.assignRelevance(cRelevance)
    }
  }

  /**
   * @param cRelevance the relevance of the candidate array
   * @param relevanceSupport the relevance support structure used during the last forward
   *
   * @return a pair containing the partitions of relevance respectively among the d1 and d2 arrays
   */
  private fun splitCandidateRelevancePartitions(cRelevance: DenseNDArray, relevanceSupport: DeltaRNNRelevanceSupport){

    val c: DenseNDArray = this.layer.candidate.valuesNotActivated
    val d2: DenseNDArray = relevanceSupport.d2.values

    relevanceSupport.d1Input.assignRelevance(
      RelevanceUtils.getRelevancePartition1(
        yRelevance = cRelevance,
        y = c,
        yContribute1 = relevanceSupport.d1Input.values,
        yContribute2 = d2,
        nPartitions = 3)
    )

    relevanceSupport.d1Rec.assignRelevance(
      RelevanceUtils.getRelevancePartition1(
        yRelevance = cRelevance,
        y = c,
        yContribute1 = relevanceSupport.d1Rec.values,
        yContribute2 = d2,
        nPartitions = 3)
    )

    relevanceSupport.d2.assignRelevance(
      RelevanceUtils.getRelevancePartition2(
        yRelevance = cRelevance,
        y = c,
        yContribute2 = d2,
        nPartitions = 3)
    )
  }

  /**
   * If n is the number of columns of this [DenseNDArray], [a] / n is added to each column of this.
   *
   * @param a the [DenseNDArray] column vector to add to this
   *
   * @return this [DenseNDArray]
   */
  private fun DenseNDArray.partialAssignSum(a: DenseNDArray): DenseNDArray {

    val aPart: DenseNDArray = a.div(this.columns.toDouble())

    for (i in 0 until this.rows) {
      val aPartI: Double = aPart[i]

      for (j in 0 until this.columns) {
        this[i, j] += aPartI
      }
    }

    return this
  }
}
