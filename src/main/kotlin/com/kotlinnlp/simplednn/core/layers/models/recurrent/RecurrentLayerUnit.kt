/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.layers.models.LayerUnit
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceUtils
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The basic unit of the recurrent layer, which extends the [LayerUnit] with the recurrent contribution.
 */
class RecurrentLayerUnit<InputNDArrayType : NDArray<InputNDArrayType>>(size: Int) : LayerUnit<InputNDArrayType>(size) {

  init {
    this.assignValues(DenseNDArrayFactory.emptyArray(Shape(size)))
  }

  /**
   * Add the recurrent contribution to the unit.
   *
   * @param parameters the parameters associated to this unit
   * @param prevContribution the output array to add as contribution from the previous state
   *
   * g += wRec (dot) prevContribution
   */
  fun addRecurrentContribution(parameters: RecurrentParametersUnit, prevContribution: DenseNDArray) {

    val wRec = parameters.recurrentWeights.values

    this.values.assignSum(wRec.dot(prevContribution))
  }

  /**
   * Assign errors to the parameters associated to this unit. The errors of the output must be already set.
   *
   * gb = errors * 1
   * gw = errors (dot) x
   * gwRec = errors (dot) yPrev
   *
   * @param paramsErrors the parameters errors associated to this unit
   * @param x the input of the unit
   * @param yPrev the output array as contribution from the previous state
   */
  fun assignParamsGradients(paramsErrors: RecurrentParametersUnit,
                            x: InputNDArrayType,
                            yPrev: DenseNDArray? = null) {

    super.assignParamsGradients(paramsErrors = paramsErrors, x = x)

    val gwRec: NDArray<*> = paramsErrors.recurrentWeights.values

    if (yPrev != null)
      gwRec.assignDot(this.errors, yPrev.t)
    else
      gwRec.zeros()
  }

  /**
   * Get the errors of the output in the previous state. The errors of the output in the current state must be
   * already set.
   *
   * @param parameters the parameters associated to this unit
   *
   * @return the errors of the recursion of this unit
   */
  fun getRecurrentErrors(parameters: RecurrentParametersUnit): DenseNDArray {

    val wRec: DenseNDArray = parameters.recurrentWeights.values as DenseNDArray

    return this.errors.t.dot(wRec)
  }

  /**
   * Get the relevance of the input of the unit. The relevance of the output must be already set.
   *
   * @param x the input of the unit
   * @param contributions the contribution of the input to calculate the output
   *
   * @return the relevance of the input of the unit
   */
  fun getInputRelevance(x: InputNDArrayType,
                        contributions: RecurrentParametersUnit,
                        prevStateExists: Boolean): NDArray<*> {

    return if (prevStateExists)
      this.getInputRelevancePartitioned(x = x, contributions = contributions)
    else
      this.getInputRelevance(x = x, contributions = contributions)
  }

  /**
   * @param x the input of the unit
   * @param contributions the contributions of this unit in the last forward
   *
   * @return the relevance of the input of the unit, calculated from the input partition of the output relevance
   */
  private fun getInputRelevancePartitioned(x: InputNDArrayType, contributions: RecurrentParametersUnit): NDArray<*> {

    val y: DenseNDArray = this.valuesNotActivated
    val yRec: DenseNDArray = contributions.biases.values as DenseNDArray
    val yInput: DenseNDArray = y.sub(yRec)
    val yInputRelevance: DenseNDArray = RelevanceUtils.getRelevancePartition1(
      yRelevance = this.relevance as DenseNDArray,
      y = y,
      yContribute1 = yInput,
      yContribute2 = yRec)  // the recurrent contrib. to y is saved into the biases contrib.

    return RelevanceUtils.calculateRelevanceOfArray(
      x = x,
      y = yInput,
      yRelevance = yInputRelevance,
      contributions = contributions.weights.values
    )
  }

  /**
   * Get the relevance of the output in the previous state. The relevance of the output in the current state must be
   * already set.
   *
   * @param contributions the contributions of this unit in the last forward
   * @param yPrev the output array as contribution from the previous state
   *
   * @return the relevance of the output in the previous state in respect of the current one
   */
  fun getRecurrentRelevance(contributions: RecurrentParametersUnit, yPrev: DenseNDArray): DenseNDArray {

    val yRec: DenseNDArray = contributions.biases.values as DenseNDArray
    val yRecRelevance: DenseNDArray = RelevanceUtils.getRelevancePartition2(
      yRelevance = this.relevance as DenseNDArray,
      y = this.valuesNotActivated,
      yContribute2 = yRec)

    return RelevanceUtils.calculateRelevanceOfDenseArray(
      x = yPrev,
      y = yRec,
      yRelevance = yRecRelevance,
      contributions = contributions.recurrentWeights.values as DenseNDArray
    )
  }
}
