/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam

import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * The Rectified ADAM method.
 *
 * @property stepSize the initial step size
 * @property beta1 the beta1 hyper-parameter
 * @property beta2 the beta2 hyper-parameter
 * @property epsilon the epsilon hyper-parameter
 * @property regularization a parameters regularization method
 */
class RADAMMethod(
  stepSize: Double = 0.001,
  beta1: Double = 0.9,
  beta2: Double = 0.999,
  epsilon: Double = 1.0E-8,
  regularization: ParamsRegularization? = null
) : ADAMMethod(
  stepSize = stepSize,
  beta1 = beta1,
  beta2 = beta2,
  epsilon = epsilon,
  regularization = regularization
) {

  /**
   * Build a [RADAMMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.RADAMConfig) : this(
    stepSize = config.stepSize,
    beta1 = config.beta1,
    beta2 = config.beta2,
    epsilon = config.epsilon,
    regularization = config.regularization
  )

  /**
   * The maximum length of the approximated SMA.
   */
  private val roMax: Double = 2.0 / (1 - this.beta2) - 1.0

  /**
   * @return the `alpha` coefficient
   */
  override fun calcAlpha(): Double {

    val b1T: Double = this.beta1.pow(this.timeStep)
    val b2T: Double = this.beta2.pow(this.timeStep)
    val ro: Double = this.roMax - 2.0 * this.timeStep * b2T / (1.0 - b2T)

    val rect: Double = if (ro > 4.0)
      sqrt((ro - 4.0) * (ro - 2.0) * this.roMax / ((this.roMax - 4.0) * (this.roMax - 2.0) * ro))
    else
      1.0

    return this.stepSize * rect * sqrt(1.0 - b2T) / (1.0 - b1T)
  }
}
