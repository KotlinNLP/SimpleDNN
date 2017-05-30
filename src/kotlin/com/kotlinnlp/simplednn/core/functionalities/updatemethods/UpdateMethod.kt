/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * UpdateMethod implements different gradient-based optimization algorithm (e.g. LearningRate, Adagrad, ADAM).
 *
 * @property regularization
 */
abstract class UpdateMethod(val regularization: WeightsRegularization?) {

  /**
   *
   * @param array parameter
   * @return the update neuralnetwork for the parameter
   */
  fun getSupportStructure(array: UpdatableDenseArray): UpdaterSupportStructure {

    if (array.updaterSupportStructure == null) {
      array.updaterSupportStructure = this.supportStructureFactory(array.values.shape)
    }

    require(isSupportStructureCompatible(array.updaterSupportStructure!!)){
      "Incompatible updaterSupportStructure"
    }

    return array.updaterSupportStructure!!
  }

  /**
   *
   * @param array the inputArray to update
   * @param errors errors to subtract to the inputArray, after being optimized
   */
  fun <NDArrayType: NDArray<NDArrayType>> update(array: UpdatableDenseArray, errors: NDArrayType) {

    val optimizedErrors: NDArrayType = this.optimizeErrors(errors, array)

    this.regularization?.apply(array)

    array.values.assignSub(optimizedErrors)
  }

  /**
   * Optimize the errors.
   *
   * @param errors the errors to optimize (sparse or dense)
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized errors
   */
  protected fun <NDArrayType: NDArray<NDArrayType>> optimizeErrors(
    errors: NDArrayType,
    array: UpdatableDenseArray
  ): NDArrayType = when (errors) {

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

  /**
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized sparse errors
   */
  abstract protected fun optimizeSparseErrors(errors: SparseNDArray, array: UpdatableDenseArray): SparseNDArray

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized dense errors
   */
  abstract protected fun optimizeDenseErrors(errors: DenseNDArray, array: UpdatableDenseArray): DenseNDArray

  /**
   *
   * @param shape shape
   * @return a new support structure to update an array
   */
  abstract protected fun supportStructureFactory(shape: Shape): UpdaterSupportStructure

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  abstract protected fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean
}
