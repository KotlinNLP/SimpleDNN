/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.*
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
   * Optimize the errors.
   *
   * @param errors the errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized errors
   */
  override fun <NDArrayType: NDArray<NDArrayType>> optimizeErrors(
    errors: NDArrayType,
    array: UpdatableDenseArray
  ): NDArrayType {

    return when (errors) {

      is SparseNDArray -> { // errors are Sparse when the input is SparseBinary
        @Suppress("UNCHECKED_CAST")
        this.optimizeSparseErrors(errors = errors, array = array) as NDArrayType
      }

      is DenseNDArray -> { // errors are Dense when the input is Dense
        @Suppress("UNCHECKED_CAST")
        this.optimizeDenseErrors(errors = errors, array = array) as NDArrayType
      }

      else -> throw RuntimeException("Invalid errors type")
    }
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

  /**
   * Optimize sparse errors.
   *
   * @param errors the sparse errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized sparse errors
   */
  private fun optimizeSparseErrors(errors: SparseNDArray, array: UpdatableDenseArray): SparseNDArray {

    val helperStructure = this.getSupportStructure(array) as ADAMStructure
    val v = helperStructure.firstOrderMoments
    val m = helperStructure.secondOrderMoments
    val mask = errors.mask

    v.assignProd(this.beta1, mask = mask).assignSum(errors.prod(1.0 - this.beta1))
    m.assignProd(this.beta2, mask = mask).assignSum(errors.prod(errors).assignProd(1.0 - this.beta2))

    return v.div(m.sqrt(mask = mask).assignSum(this.epsilon), mask = mask).assignProd(this.alpha)
  }

  /**
   * Optimize dense errors.
   *
   * @param errors the dense errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized dense errors
   */
  private fun optimizeDenseErrors(errors: DenseNDArray, array: UpdatableDenseArray): DenseNDArray {

    val helperStructure = this.getSupportStructure(array) as ADAMStructure
    val v = helperStructure.firstOrderMoments
    val m = helperStructure.secondOrderMoments

    v.assignProd(this.beta1).assignSum(errors.prod(1.0 - this.beta1))
    m.assignProd(this.beta2).assignSum(errors.prod(errors).assignProd(1.0 - this.beta2))

    return v.div(m.sqrt().assignSum(this.epsilon)).assignProd(this.alpha)
  }
}
