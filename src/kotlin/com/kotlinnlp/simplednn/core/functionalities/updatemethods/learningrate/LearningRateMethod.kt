/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.ExponentialDecay
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
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
   *
   * @param errors errors
   * @return optimized errors
   */
  override fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray {

    val helperStructure = this.getSupportStructure(array) as LearningRateStructure

    helperStructure.errors.assignProd(errors, this.alpha)

    return helperStructure.errors
  }

  /**
   * Method to call every new epoch
   */
  override fun newEpoch() {

    this.epochCount += 1

    when(decayMethod){
      is HyperbolicDecay ->
        this.alpha = decayMethod.update(learningRate, this.epochCount)

      is ExponentialDecay ->
        this.alpha = decayMethod.update(this.alpha, this.epochCount)
    }
  }
}
