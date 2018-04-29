/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import kotlin.reflect.KClass

/**
 * The LearningRate method.
 *
 * @param learningRate learningRate
 * @param decayMethod decayMethod
 */
class LearningRateMethod(
  val learningRate: Double,
  val decayMethod: DecayMethod? = null,
  regularization: WeightsRegularization? = null
) : EpochScheduling,
    UpdateMethod<LearningRateStructure>(regularization, LearningRateStructure::class) {

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
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized sparse errors
   */
  override fun optimizeSparseErrors(errors: SparseNDArray, array: UpdatableDenseArray): SparseNDArray =
    errors.prod(this.alpha)

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param array an [UpdatableDenseArray]
   *
   * @return optimized dense errors
   */
  override fun optimizeDenseErrors(errors: DenseNDArray, array: UpdatableDenseArray): DenseNDArray {

    val helperStructure: LearningRateStructure = this.getSupportStructure(array)
    helperStructure.errors.assignProd(errors, this.alpha)

    return helperStructure.errors
  }
}
