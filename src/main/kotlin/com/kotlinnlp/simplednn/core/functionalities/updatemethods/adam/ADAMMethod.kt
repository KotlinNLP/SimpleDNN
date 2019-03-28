/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The ADAM method.
 *
 * @param stepSize stepSize
 * @param beta1 beta1
 * @param beta2 beta2
 * @param epsilon epsilon
 */
class ADAMMethod(
  val stepSize: Double = 0.001,
  val beta1: Double = 0.9,
  val beta2: Double = 0.999,
  val epsilon: Double = 1.0E-8,
  regularization: WeightsRegularization? = null
) : ExampleScheduling,
    UpdateMethod<ADAMStructure>(regularization) {

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: UpdatableDenseArray): ADAMStructure = array.getOrSetSupportStructure()

  /**
   * The 'alpha' coefficient.
   */
  var alpha: Double = this.stepSize
    private set

  /**
   * The number of examples seen.
   */
  private var exampleCount: Double = 0.0

  /**
   * Method to call every new example
   */
  override fun newExample() {
    this.exampleCount++
    this.updateAlpha()
  }

  /**
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized sparse errors
   */
  override fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: ADAMStructure): SparseNDArray {

    val v = supportStructure.firstOrderMoments
    val m = supportStructure.secondOrderMoments
    val mask = errors.mask

    v.assignProd(this.beta1, mask = mask).assignSum(errors.prod(1.0 - this.beta1))
    m.assignProd(this.beta2, mask = mask).assignSum(errors.prod(errors).assignProd(1.0 - this.beta2))

    return v.div(m.sqrt(mask = mask).assignSum(this.epsilon)).assignProd(this.alpha)
  }

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  override fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: ADAMStructure): DenseNDArray {

    val v = supportStructure.firstOrderMoments
    val m = supportStructure.secondOrderMoments

    v.assignProd(this.beta1).assignSum(errors.prod(1.0 - this.beta1))
    m.assignProd(this.beta2).assignSum(errors.prod(errors).assignProd(1.0 - this.beta2))

    return v.div(m.sqrt().assignSum(this.epsilon)).assignProd(this.alpha)
  }

  /**
   *
   */
  private fun updateAlpha() {
    this.alpha = this.stepSize *
      Math.sqrt(1.0 - Math.pow(this.beta2, this.exampleCount)) /
      (1.0 - Math.pow(this.beta1, this.exampleCount))
  }
}
