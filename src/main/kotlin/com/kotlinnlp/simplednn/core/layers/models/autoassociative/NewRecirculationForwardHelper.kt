/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.autoassociative

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a NewRecirculation layer.
 *
 * @property layer the [NewRecirculationLayer] in which the forward is executed
 */
internal class NewRecirculationForwardHelper(
  override val layer: NewRecirculationLayer
) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *   yR = f(w (dot) xR + b)
   *   xI = r * xR + (1 - r) * w' (dot) yR    // outputArray
   *   yI = r * yR + (1 - r) * f(w (dot) xI)
   */
  override fun forward() {

    val r: Double = this.layer.lambda
    val w: DenseNDArray = this.layer.params.unit.weights.values
    val b: DenseNDArray = this.layer.params.unit.biases.values
    val xR: DenseNDArray = this.layer.realInput.values
    val yR: DenseNDArray = this.layer.realOutput.values
    val xI: DenseNDArray = this.layer.imaginaryInput.values
    val yI: DenseNDArray = this.layer.imaginaryOutput.values

    yR.assignDot(w, xR).assignSum(b)
    this.layer.realOutput.activate()

    // Note of optimization: double transposition of two 1-dim arrays instead of a bigger 2-dim one
    xI.assignSum(xR.prod(r), yR.t.dot(w).t.assignProd(1 - r))

    yI.assignDot(w, xI).assignSum(b)
    this.layer.imaginaryOutput.activate()

    yI.assignProd(1 - r).assignSum(yR.prod(r))
  }
}
