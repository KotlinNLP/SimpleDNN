/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The RMSProp method is a variant of [com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod]
 * where the squared sum of previous gradients is replaced with a moving average.
 *
 * @property learningRate Double >= 0. Initial learning rate
 * @property epsilon Double >= 0. Bias parameter
 * @property decay Learning rate decay parameter
 *
 * References
 * [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 *
 */
class RMSPropMethod(
  val learningRate: Double = 0.001,
  val epsilon: Double = 1e-08,
  val decay: Double = 0.95,
  regularization: WeightsRegularization? = null
) : UpdateMethod(regularization) {

  /**
   *
   * @param shape shape
   * @return helper update neuralnetwork
   */
  override fun supportStructureFactory(shape: Shape): UpdaterSupportStructure = RMSPropStructure(shape)

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  override fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean {
    return supportStructure is RMSPropStructure
  }

  /**
   *
   * @param errors errors
   * @return optimized errors
   */
  override fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray {

    val helperStructure = this.getSupportStructure(array) as RMSPropStructure

    val secondOrderMoments = helperStructure.secondOrderMoments

    secondOrderMoments.assignValues(secondOrderMoments.prod(decay).sum(errors.prod(errors).prod(1.0 - decay)))

    return errors.div(secondOrderMoments.sqrt().sum(epsilon)).prod(this.learningRate)
  }
}
