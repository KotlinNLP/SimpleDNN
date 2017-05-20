/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 */
class GateUnit(size: Int) : AugmentedArray(size) {

  /**
   *
   * @param gateParams the parameters of the gate
   * @param x the inputArray of the current layer
   *
   * g = w (dot) x + b
   */
  fun forward(gateParams: GateParametersUnit, x: NDArray) {

    val w = gateParams.weights.values
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
  fun addRecurrentContribute(gateParams: GateParametersUnit, prevContribute: NDArray) {

    val wRec = gateParams.recurrentWeights.values

    this.values.assignSum(wRec.dot(prevContribute))
  }
}
