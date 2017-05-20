/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The AdaGrad method assigns a different learning rate to each parameter
 * using the sum of squares of its all historical gradients.
 *
 * References
 * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
 *
 * @property learningRate Initial learning rate
 * @property epsilon Bias parameter
 * @property regularization
 */
class AdaGradMethod(
  val learningRate:Double = 0.01,
  val epsilon: Double = 1.0E-8,
  regularization: WeightsRegularization? = null
) : UpdateMethod(regularization) {

  /**
   *
   * @param shape shape
   * @return helper update neuralnetwork
   */
  override fun supportStructureFactory(shape: Shape): UpdaterSupportStructure = AdaGradStructure(shape)

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  override fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean {
    return supportStructure is AdaGradStructure
  }

  /**
   *
   * @param errors errors
   * @return optimized errors
   */
  override fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray {
    val helperStructure = this.getSupportStructure(array) as AdaGradStructure

    helperStructure.secondOrderMoments.assignSum(errors.prod(errors))

    return errors.div(helperStructure.secondOrderMoments.sqrt().sum(epsilon)).prod(this.learningRate)
  }
}
