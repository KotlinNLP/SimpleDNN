/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.nesterovmomentum

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * @param learningRate Double >= 0. Learning rate
 * @param momentum  Double >= 0. Parameter updates momentum
 */
class NesterovMomentumMethod(
  learningRate: Double = 0.01,
  momentum: Double = 0.9,
  decayMethod: DecayMethod? = null,
  regularization: WeightsRegularization? = null
): MomentumMethod(
  learningRate = learningRate,
  momentum = momentum,
  decayMethod = decayMethod,
  regularization = regularization) {

  /**
   *
   * @param shape shape
   * @return helper update neuralnetwork
   */
  override fun supportStructureFactory(shape: Shape): UpdaterSupportStructure = NesterovMomentumStructure(shape)

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  override fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean {
    return supportStructure is NesterovMomentumStructure
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
   * Optimize sparse errors.
   *
   * @param errors the sparse errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized sparse errors
   */
  private fun optimizeSparseErrors(errors: SparseNDArray, array: UpdatableDenseArray): SparseNDArray {

    val helperStructure = this.getSupportStructure(array) as NesterovMomentumStructure
    val v = helperStructure.v
    val vPrev = helperStructure.vPrev

    val mask = errors.mask

    vPrev.assignValues(v, mask = mask) // backup previous velocity

    v.assignValues(errors.prod(this.alpha).assignSum(v.prod(this.momentum, mask = mask)))

    return vPrev.prod(-this.momentum, mask = mask).assignSum(v.prod(1.0 + this.momentum, mask = mask))
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

    val helperStructure = this.getSupportStructure(array) as NesterovMomentumStructure
    val v = helperStructure.v
    val vPrev = helperStructure.vPrev

    vPrev.assignValues(v) // backup previous velocity

    v.assignSum(errors.prod(this.alpha), v.prod(this.momentum))

    return vPrev.prod(-this.momentum).assignSum(v.prod(1.0 + this.momentum))
  }
}
