/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceUtils
import com.kotlinnlp.simplednn.core.layers.types.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The helper which calculates the relevance of the input of a [layer] respect of its output.
 *
 * @property layer the [DeltaRNNLayerStructure] in which to calculate the input relevance
 */
class DeltaRNNRelevanceHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: DeltaRNNLayerStructure<InputNDArrayType>
) : GatedRecurrentRelevanceHelper<InputNDArrayType>(layer) {

  /**
   * Propagate the relevance from the output to the array units of the layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun propagateRelevanceToGates(layerContributions: LayerParameters<*>) {
    layerContributions as DeltaRNNLayerParameters

    val previousStateExists: Boolean = this.layer.layerContextWindow.getPrevStateLayer() != null

    val halfOutputRelevance: DenseNDArray = (this.layer.outputArray.relevance as DenseNDArray).div(2.0)

    val candidateRelevance: DenseNDArray = if (previousStateExists)
      this.getInputPartition(layerContributions).div(2.0)
    else
      halfOutputRelevance // if there isn't a previous state, all the output relevance is assigned to the candidate
                          // partition (p * c), half to the partition array and half to the candidate array

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
  override fun getInputRelevance(layerContributions: LayerParameters<*>): NDArray<*> {
    this.layer.params as DeltaRNNLayerParameters
    layerContributions as DeltaRNNLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values

    val wxContrib: NDArray<*> = layerContributions.feedforwardUnit.weights.values

    val relevanceSupport: DeltaRNNRelevanceSupport = this.layer.relevanceSupport
    val previousStateExists: Boolean = this.layer.layerContextWindow.getPrevStateLayer() != null

    val bp: DenseNDArray = this.layer.params.recurrentUnit.biases.values as DenseNDArray
    val bc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values as DenseNDArray
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val d1Bc: DenseNDArray = if (previousStateExists) bc.div(2.0) else bc
    // if there is a recurrent contribution bc is divided equally among d1Input and d1Rec, otherwise it is all assigned
    // to d1Input

    val pInputRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = this.layer.partition.valuesNotActivated,
      yRelevance = this.layer.partition.relevance as DenseNDArray,
      contributions = this.assignSum(wxContrib.copy(), bp) // w (dot) x + bp
    )

    val d1InputRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = relevanceSupport.d1Input.values,
      yRelevance = relevanceSupport.d1Input.relevance as DenseNDArray,
      contributions = this.assignSum(wxContrib.prod(beta1), d1Bc) // (w (dot) x) * beta1 + (bc | bc / 2)
    )

    val inputRelevance: NDArray<*> = pInputRelevance.assignSum(d1InputRelevance)

    if (previousStateExists) {
      val d2InputRelevance = RelevanceUtils.calculateRelevanceOfArray(
        x = x,
        y = this.layer.wx.values, // the product by 'alpha' is not included during the calculation of the relevance
                                  // ('alpha' doesn't depend on variables of interest)
        yRelevance = (relevanceSupport.d2.relevance as DenseNDArray).div(2.0),
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
  override fun setRecurrentRelevance(layerContributions: LayerParameters<*>) {
    this.layer.params as DeltaRNNLayerParameters
    layerContributions as DeltaRNNLayerParameters

    val prevStateOutput: AugmentedArray<DenseNDArray> = this.layer.layerContextWindow.getPrevStateLayer()!!.outputArray
    val yPrev: DenseNDArray = prevStateOutput.values

    val wyRecContrib: DenseNDArray = layerContributions.recurrentUnit.weights.values as DenseNDArray

    val halfBc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values.div(2.0) as DenseNDArray
    val beta2: DenseNDArray = this.layer.params.beta2.values

    val relevanceSupport: DeltaRNNRelevanceSupport = this.layer.relevanceSupport

    val d1RecRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = yPrev,
      y = relevanceSupport.d1Rec.values,
      yRelevance = relevanceSupport.d1Rec.relevance as DenseNDArray,
      contributions = this.assignSum(wyRecContrib.prod(beta2), halfBc) // (wyRec (dot) yPrev) * beta2 + bc / 2
    )

    val d2RecRelevance = RelevanceUtils.calculateRelevanceOfArray(
      x = yPrev,
      y = this.layer.wyRec.values, // the product by 'alpha' is not included during the calculation of the relevance
                                   // ('alpha' doesn't depend on variables of interest)
      yRelevance = (relevanceSupport.d2.relevance as DenseNDArray).div(2.0),
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
    val yRec: DenseNDArray = contributions.recurrentUnit.biases.values as DenseNDArray
    val yInput: DenseNDArray = y.sub(yRec)

    return RelevanceUtils.getRelevancePartition1(
      yRelevance = this.layer.outputArray.relevance as DenseNDArray,
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
      yRelevance = this.layer.outputArray.relevance as DenseNDArray,
      y = this.layer.outputArray.valuesNotActivated,
      yContribute2 = contributions.recurrentUnit.biases.values as DenseNDArray)
  }

  /**
   * Set the partitions of the relevance of the candidate splitting it respectively among the d1 and d2 arrays.
   * The d1 array is itself composed by an input partition and a recurrent partition.
   * If there isn't a previous state all the candidate relevance is assigned to the d1 input partition.
   */
  private fun setCandidateRelevancePartitions(previousStateExists: Boolean) {

    val cRelevance: DenseNDArray = this.layer.candidate.relevance as DenseNDArray
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
   * Special assignSum method which dispatches the call of the partialAssignSum method checking the type of [a].
   *
   * @param a the generic [NDArray] to which to add [b]
   * @param b the [DenseNDArray] to add to [a]
   *
   * @return [a] after the addition of [b]
   */
  private fun assignSum(a: NDArray<*>, b: DenseNDArray): NDArray<*> {
    require(a.rows == b.rows) { "b must be a column vector with the same number of rows of a" }
    require(b.columns == 1) { "b must be a column vector" }

    return when (a) {
      is DenseNDArray -> a.partialAssignSum(b)
      is SparseNDArray -> a.partialAssignSum(b)
      else -> throw RuntimeException("Invalid NDArray type")
    }
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

  /**
   * This method should be called if this [SparseNDArray] contains sparse columns.
   * If n is the number of sparse columns, [a] / n is added to each active column of this.
   *
   * @param a the [DenseNDArray] column vector to add to this
   *
   * @return this [SparseNDArray]
   */
  private fun SparseNDArray.partialAssignSum(a: DenseNDArray): SparseNDArray {

    val activeColumns: Int = this.colIndices.toSet().size

    val aPart: DenseNDArray = a.div(activeColumns.toDouble())

    for (k in 0 until this.values.size) {
      val i: Int = this.rowIndices[k]
      this.values[k] += aPart[i]
    }

    return this
  }
}
