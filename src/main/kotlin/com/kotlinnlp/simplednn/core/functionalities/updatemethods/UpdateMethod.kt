/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * UpdateMethod implements different gradient-based optimization algorithm (e.g. LearningRate, Adagrad, ADAM).
 *
 * @property regularization
 */
abstract class UpdateMethod<SupportStructureType: UpdaterSupportStructure>(
  val regularization: WeightsRegularization?
) {

  /**
   * Update the given [array] with the given [errors].
   *
   * @param array the inputArray to update
   * @param errors errors to subtract to the inputArray, after being optimized
   */
  fun <NDArrayType: NDArray<NDArrayType>> update(array: ParamsArray, errors: NDArrayType) {

    val optimizedErrors: NDArray<*> = this.optimize(errors, array)

    this.regularization?.apply(array)

    array.values.assignSub(optimizedErrors)
  }

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  abstract fun getSupportStructure(array: ParamsArray): SupportStructureType

  /**
   * Optimize the errors.
   *
   * @param errors the errors to optimize (sparse or dense)
   * @param array the [ParamsArray]
   *
   * @return optimized errors
   */
  private fun <NDArrayType: NDArray<NDArrayType>> optimizeErrors(
    errors: NDArrayType,
    array: ParamsArray
  ): NDArrayType = when (errors) {

    // errors are Sparse when the input is SparseBinary
    is SparseNDArray -> this.optimizeSparseErrors(errors, this.getSupportStructure(array))

    // errors are Dense when the input is Dense
    is DenseNDArray -> this.optimizeDenseErrors(errors, this.getSupportStructure(array))

    else -> throw RuntimeException("Invalid errors type")
  }

  /**
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized sparse errors
   */
  protected open fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: SupportStructureType): NDArray<*> =
    this.optimizeGenericErrors(errors = errors, supportStructure = supportStructure)

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  protected open fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: SupportStructureType): NDArray<*> =
    this.optimizeGenericErrors(errors = errors, supportStructure = supportStructure)

  /**
   * Optimize generic errors.
   *
   * @param errors the generic errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized generic errors
   */
  protected open fun optimizeGenericErrors(errors: NDArray<*>, supportStructure: SupportStructureType): NDArray<*> {
    throw NotImplementedError("The method 'optimizeGenericErrors' must be implemented if 'optimizeSparseErrors' and" +
      "'optimizeDenseErrors' are not overridden.")
  }
}
