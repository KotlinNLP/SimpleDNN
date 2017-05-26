/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling

/**
 *
 * @param learningRate learningRate
 * @param decayMethod decayMethod
 */
class LearningRateMethod(
  val learningRate: Double,
  val decayMethod: DecayMethod? = null,
  regularization: WeightsRegularization? = null
) : EpochScheduling,
    UpdateMethod(regularization) {

  /**
   *
   */
  var alpha: Double = this.learningRate
    private set

  /**
   *
   */
  private var epochCount: Int = 0

  /**
   *
   * @param shape shape
   * @return helper update neuralnetwork
   */
  override fun supportStructureFactory(shape: Shape): UpdaterSupportStructure = LearningRateStructure(shape)

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  override fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean {
    return supportStructure is LearningRateStructure
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
        errors.prod(this.alpha) as NDArrayType
      }

      is DenseNDArray -> { // errors are Dense when the input is Dense
        @Suppress("UNCHECKED_CAST")
        this.optimizeDenseErrors(errors = errors, array = array) as NDArrayType
      }

      else -> throw RuntimeException("Invalid errors type")
    }
  }

  /**
   * Method to call every new epoch
   */
  override fun newEpoch() {

    if (this.decayMethod != null) {
      this.alpha = this.decayMethod.update(
        learningRate = if (this.decayMethod is HyperbolicDecay) this.learningRate else this.alpha,
        timeStep = ++this.epochCount
      )
    }
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

    val helperStructure = this.getSupportStructure(array) as LearningRateStructure
    helperStructure.errors.assignProd(errors, this.alpha)

    return helperStructure.errors
  }
}
