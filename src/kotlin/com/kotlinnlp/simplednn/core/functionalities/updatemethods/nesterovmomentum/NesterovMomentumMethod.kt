/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.nesterovmomentum

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

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
   *
   * @param errors errors
   * @return optimized errors
   */
  override fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray {

    val helperStructure = this.getSupportStructure(array) as NesterovMomentumStructure

    helperStructure.vPrev.assignValues(helperStructure.v) // backup previous velocity

    // update velocity with adapted learning rates
    helperStructure.v.assignValues(errors.prod(this.alpha).sum(helperStructure.v.prod(this.momentum)))

    return helperStructure.vPrev.prod(-this.momentum).sum(helperStructure.v.prod(1.0 + this.momentum))
  }
}
