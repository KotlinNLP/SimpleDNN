/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling

/**
 * The Momentum method.
 *
 * @property learningRate the initial learning rate
 * @property momentum  the momentum
 * @property decayMethod the rate decay method
 * @property regularization a parameters regularization method
 */
class MomentumMethod(
  val learningRate: Double = 0.01,
  val momentum: Double = 0.9,
  val decayMethod: DecayMethod? = null,
  regularization: ParamsRegularization? = null
) : EpochScheduling,
  UpdateMethod<MomentumStructure>(regularization) {

  /**
   * Build a [MomentumMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.MomentumConfig): this(
    learningRate = config.learningRate,
    momentum = config.momentum,
    decayMethod = config.decayMethod,
    regularization = config.regularization
  )

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): MomentumStructure = array.getOrSetSupportStructure()

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
  override fun optimizeGenericErrors(errors: NDArray<*>, supportStructure: MomentumStructure): DenseNDArray =
    supportStructure.v.assignProd(this.momentum).assignSum(errors.prod(this.alpha))
}
