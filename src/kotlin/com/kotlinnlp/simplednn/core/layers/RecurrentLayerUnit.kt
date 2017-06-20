/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

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
   * Assign errors to the parameters associated to this unit.
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

    if (yPrev != null) {
      val gwRec: DenseNDArray = paramsErrors.recurrentWeights.values
      gwRec.assignDot(this.errors, yPrev.T)
    }
  }
}
