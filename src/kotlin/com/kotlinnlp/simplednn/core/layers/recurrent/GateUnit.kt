/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
class GateUnit<InputNDArrayType : NDArray<InputNDArrayType>>(size: Int) : AugmentedArray<DenseNDArray>(size) {

  init {
    this.assignValues(DenseNDArrayFactory.emptyArray(Shape(size)))
  }

  /**
   *
   * @param gateParams the parameters of the gate
   * @param x the inputArray of the current layer
   *
   * g = w (dot) x + b
   */
  fun forward(gateParams: GateParametersUnit, x: InputNDArrayType) {

    val w = gateParams.weights.values as DenseNDArray
    val b = gateParams.biases.values

    this.values.assignDot(w, x).assignSum(b)
  }

  /**
   *
   * @param gateParams the parameters of the gate of the next layer
   * @param prevContribute the inputArray to add as contribute from the previous state
   *
   * g += wRec (dot) prevContribute
   */
  fun addRecurrentContribute(gateParams: GateParametersUnit, prevContribute: DenseNDArray) {

    val wRec = gateParams.recurrentWeights.values

    this.values.assignSum(wRec.dot(prevContribute))
  }

  /**
   * Assign errors gradients to the [paramsErrors] associated to this gate
   *
   * @param paramsErrors a [GateParametersUnit] associated to this gate
   * @param x the input [NDArray] of the gate
   * @param yPrev the output [NDArray] of the gate in the previous state
   *
   * gb = gGate * 1
   * gw = gGate (dot) x
   * gwRec = gGate (dot) yPrev
   */
  fun assignParamsGradients(paramsErrors: GateParametersUnit,
                            x: InputNDArrayType,
                            yPrev: DenseNDArray? = null) {

    val gGate: DenseNDArray = this.errors
    val gb: DenseNDArray = paramsErrors.biases.values
    val gw: NDArray<*> = paramsErrors.weights.values
    val gwRec: DenseNDArray = paramsErrors.recurrentWeights.values

    gb.assignValues(gGate)
    gw.assignDot(gGate, x.T)

    if (yPrev != null) {
      gwRec.assignDot(gGate, yPrev.T)
    }
  }
}
