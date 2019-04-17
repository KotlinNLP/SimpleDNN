/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.nesterovmomentum.NesterovMomentumMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop.RMSPropMethod
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

  companion object {

    /**
     * @param config the configuration
     *
     * @return the [UpdateMethod] of the given [config]
     */
    operator fun invoke(config: UpdateMethodConfig) = when(config) {
      is UpdateMethodConfig.AdaGradConfig -> AdaGradMethod(config)
      is UpdateMethodConfig.ADAMConfig -> ADAMMethod(config)
      is UpdateMethodConfig.LearningRateConfig -> LearningRateMethod(config)
      is UpdateMethodConfig.MomentumConfig -> MomentumMethod(config)
      is UpdateMethodConfig.NesterovMomentumConfig -> NesterovMomentumMethod(config)
      is UpdateMethodConfig.RMSPropConfig -> RMSPropMethod(config)
    }
  }

  /**
   * Update the given [array] with the given [errors].
   *
   * @param array the inputArray to update
   * @param errors errors to subtract to the inputArray, after being optimized
   */
  fun update(array: ParamsArray, errors: NDArray<*>) {

    val optimizedErrors: NDArray<*> = this.optimizeErrors(errors, array)

    this.regularization?.apply(array)

    array.values.assignSub(optimizedErrors)
  }

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  internal abstract fun getSupportStructure(array: ParamsArray): SupportStructureType

  /**
   * Optimize the errors.
   *
   * @param errors the errors to optimize (sparse or dense)
   * @param array a [ParamsArray]
   *
   * @return optimized errors
   */
  private fun optimizeErrors(errors: NDArray<*>, array: ParamsArray): NDArray<*> = when (errors) {

    is SparseNDArray -> // errors are Sparse when the input is SparseBinary
      this.optimizeSparseErrors(errors, this.getSupportStructure(array))

    is DenseNDArray -> // errors are Dense when the input is Dense
      this.optimizeDenseErrors(errors, this.getSupportStructure(array))

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
