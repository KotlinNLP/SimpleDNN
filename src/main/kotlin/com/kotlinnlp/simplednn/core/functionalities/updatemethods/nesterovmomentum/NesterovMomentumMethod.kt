/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.nesterovmomentum

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling

/**
 * The NesterovMomentum method.
 *
 * @param learningRate Double >= 0. Learning rate
 * @param momentum  Double >= 0. Parameter updates momentum
 */
class NesterovMomentumMethod(
  val learningRate: Double = 0.01,
  val momentum: Double = 0.9,
  val decayMethod: DecayMethod? = null,
  regularization: WeightsRegularization? = null
) : EpochScheduling, UpdateMethod<NesterovMomentumStructure>(regularization) {

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): NesterovMomentumStructure =
    array.getOrSetSupportStructure()

  /**
   * The 'alpha' coefficient.
   */
  var alpha: Double = this.learningRate
    private set

  /**
   * The number of epochs seen.
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
   * Optimize errors.
   *
   * @param errors the errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return the optimized errors
   */
  override fun optimizeGenericErrors(errors: NDArray<*>, supportStructure: NesterovMomentumStructure): DenseNDArray {

    val v = supportStructure.v
    val vTmp = supportStructure.vTmp

    vTmp.assignProd(v, this.momentum)

    v.assignValues(vTmp).assignSum(errors.prod((this.alpha)))

    return v.prod(1.0 + this.momentum).assignSub(vTmp)
  }
}
