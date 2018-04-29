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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import kotlin.reflect.KClass

/**
 * UpdateMethod implements different gradient-based optimization algorithm (e.g. LearningRate, Adagrad, ADAM).
 *
 * @property regularization
 * @param structureClass the Kotlin Class of the support structure of this updater
 */
abstract class UpdateMethod<SupportStructureType: UpdaterSupportStructure>(
  val regularization: WeightsRegularization?,
  private val structureClass: KClass<SupportStructureType>
) {

  /**
   * Update the given [array] with the given [errors].
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
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  internal fun getSupportStructure(array: UpdatableDenseArray): SupportStructureType {

    try {
      array.updaterSupportStructure
    } catch (e: UninitializedPropertyAccessException) {
      array.updaterSupportStructure = this.structureFactory(array)
    }

    require(this.structureClass.isInstance(array.updaterSupportStructure)) { "Incompatible updaterSupportStructure" }

    @Suppress("UNCHECKED_CAST")
    return array.updaterSupportStructure as SupportStructureType
  }

  /**
   * @param array the array to update
   *
   * @return a new [UpdaterSupportStructure] to associate to the [array]
   */
  private fun structureFactory(array: UpdatableDenseArray): SupportStructureType
    = this.structureClass.constructors.first().call(array.values.shape)

  /**
   * Optimize the errors.
   *
   * @param errors the errors to optimize (sparse or dense)
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized errors
   */
  private fun <NDArrayType: NDArray<NDArrayType>> optimizeErrors(
    errors: NDArrayType,
    array: UpdatableDenseArray
  ): NDArrayType = when (errors) {

    is SparseNDArray -> { // errors are Sparse when the input is SparseBinary
      @Suppress("UNCHECKED_CAST")
      this.optimizeSparseErrors(errors, this.getSupportStructure(array)) as NDArrayType
    }

    is DenseNDArray -> { // errors are Dense when the input is Dense
      @Suppress("UNCHECKED_CAST")
      this.optimizeDenseErrors(errors, this.getSupportStructure(array)) as NDArrayType
    }

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
  protected abstract fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: SupportStructureType): SparseNDArray

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  protected abstract fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: SupportStructureType): DenseNDArray
}
