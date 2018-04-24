/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The basic unit of the recurrent layer, which extends the [LayerUnit] with the recurrent contribution.
 */
class IndRNNLayerUnit<InputNDArrayType : NDArray<InputNDArrayType>>(size: Int) : LayerUnit<InputNDArrayType>(size) {

  init {
    this.assignValues(DenseNDArrayFactory.emptyArray(Shape(size)))
  }

  /**
   * Add the recurrent contribution to the unit.
   *
   * @param parameters the parameters associated to this unit
   * @param prevContribution the output array to add as contribution from the previous state
   *
   * g += wRec * prevContribution
   */
  fun addRecurrentContribution(parameters: IndRNNParametersUnit, prevContribution: DenseNDArray) {

    val wRec = parameters.recurrentWeights.values

    this.values.assignSum(wRec.prod(prevContribution))
  }

  /**
   * Assign errors to the parameters associated to this unit. The errors of the output must be already set.
   *
   * gb = errors * 1
   * gw = errors (dot) x
   * gwRec = errors * yPrev
   *
   * @param paramsErrors the parameters errors associated to this unit
   * @param x the input of the unit
   * @param yPrev the output array as contribution from the previous state
   * @param mePropMask the mask of the top k output nodes, in order to execute the 'meProp' algorithm
   */
  fun assignParamsGradients(paramsErrors: IndRNNParametersUnit,
                            x: InputNDArrayType,
                            yPrev: DenseNDArray? = null,
                            mePropMask: NDArrayMask? = null) {

    // TODO: mePropMask

    super.assignParamsGradients(paramsErrors = paramsErrors, x = x, mePropMask = mePropMask)

    val gwRec = paramsErrors.recurrentWeights.values as DenseNDArray

    if (yPrev != null) {

      gwRec.assignProd(this.errors, yPrev)

    } else {
      gwRec.zeros()
    }
  }

  /**
   * Get the errors of the output in the previous state. The errors of the output in the current state must be
   * already set.
   *
   * @param parameters the parameters associated to this unit
   * @param mePropMask the mask of the top k output nodes, in order to execute the 'meProp' algorithm
   *
   * @return the errors of the recursion of this unit
   */
  fun getRecurrentErrors(parameters: IndRNNParametersUnit, mePropMask: NDArrayMask? = null): DenseNDArray {

    // TODO: mePropMask

    val wRec: DenseNDArray = parameters.recurrentWeights.values as DenseNDArray

    return this.errors.prod(wRec)
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
                        contributions: IndRNNParametersUnit,
                        prevStateExists: Boolean): NDArray<*> {

    TODO()
  }

  /**
   * @param x the input of the unit
   * @param contributions the contributions of this unit in the last forward
   *
   * @return the relevance of the input of the unit, calculated from the input partition of the output relevance
   */
  private fun getInputRelevancePartitioned(x: InputNDArrayType, contributions: IndRNNParametersUnit): NDArray<*> {

    TODO()
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
  fun getRecurrentRelevance(contributions: IndRNNParametersUnit, yPrev: DenseNDArray): DenseNDArray {

    TODO()
  }
}
