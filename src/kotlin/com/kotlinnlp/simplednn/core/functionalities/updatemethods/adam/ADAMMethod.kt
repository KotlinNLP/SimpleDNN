/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 *
 * @param stepSize stepSize
 * @param beta1 beta1
 * @param beta2 beta2
 * @param epsilon epsilon
 */
class ADAMMethod(
  val stepSize:Double = 0.001,
  val beta1: Double = 0.9,
  val beta2: Double = 0.9,
  val epsilon: Double = 1.0E-8,
  regularization: WeightsRegularization? = null
) : ExampleScheduling,
    UpdateMethod(regularization) {

  /**
   *
   */
  var alpha: Double = this.stepSize
    private set

  /**
   *
   */
  private var exampleCount: Double = 0.0

  /**
   *
   * @param shape shape
   * @return helper update neuralnetwork
   */
  override fun supportStructureFactory(shape: Shape): UpdaterSupportStructure = ADAMStructure(shape)

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  override fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean {
    return supportStructure is ADAMStructure
  }

  /**
   *
   * @param errors errors
   * @return optimized errors
   */
  override fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray {
    val helperStructure = this.getSupportStructure(array) as ADAMStructure

    helperStructure.firstOrderMoments.assignProd(this.beta1).assignSum(errors.prod(1.0 - this.beta1))
    helperStructure.secondOrderMoments.assignProd(this.beta2).assignSum(
      errors.prod(errors).assignProd(1.0 - this.beta2))

    return helperStructure.firstOrderMoments.div(
      helperStructure.secondOrderMoments.sqrt().assignSum(this.epsilon)).assignProd(this.alpha)
  }

  /**
   * Method to call every new example
   */
  override fun newExample() {
    this.exampleCount++
    this.updateAlpha()
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
