/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * The ADAM method.
 *
 * @property stepSize the initial step size
 * @property beta1 the beta1 hyper-parameter
 * @property beta2 the beta2 hyper-parameter
 * @property epsilon the epsilon hyper-parameter
 * @property regularization a parameters regularization method
 */
open class ADAMMethod(
  val stepSize: Double = 0.001,
  val beta1: Double = 0.9,
  val beta2: Double = 0.999,
  val epsilon: Double = 1.0E-8,
  regularization: ParamsRegularization? = null
) : BatchScheduling, UpdateMethod<ADAMStructure>(regularization) {

  /**
   * Build an [ADAMMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.ADAMConfig): this(
    stepSize = config.stepSize,
    beta1 = config.beta1,
    beta2 = config.beta2,
    epsilon = config.epsilon,
    regularization = config.regularization
  )

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): ADAMStructure = array.getOrSetSupportStructure()

  /**
   * The current time step.
   */
  protected var timeStep: Int = 1

  /**
   * Check requirements.
   */
  init {
    require(this.beta1 >= 0.0 && this.beta1 < 1.0) { "`beta1` must be in the range [0.0, 1.0)" }
    require(this.beta2 >= 0.0 && this.beta2 < 1.0) { "`beta2` must be in the range [0.0, 1.0)" }
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.timeStep++
  }

  /**
   * Optimize errors.
   *
   * @param errors the errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return the optimized errors
   */
  override fun optimizeGenericErrors(errors: NDArray<*>, supportStructure: ADAMStructure): DenseNDArray {

    val m: DenseNDArray = supportStructure.firstOrderMoments
    val v: DenseNDArray = supportStructure.secondOrderMoments

    val sqrtB2T: Double = sqrt(1.0 - this.beta2.pow(this.timeStep))
    val alpha: Double = this.calcAlpha()

    m.assignProd(this.beta1).assignSum(errors.prod(1.0 - this.beta1))
    v.assignProd(this.beta2).assignSum(errors.prod(errors).assignProd(1.0 - this.beta2))

    return m.div(v.sqrt().assignSum(this.epsilon * sqrtB2T)).assignProd(alpha)
  }

  /**
   * @return the `alpha` coefficient
   */
  protected open fun calcAlpha(): Double =
    this.stepSize * sqrt(1.0 - this.beta2.pow(this.timeStep)) / (1.0 - this.beta1.pow(this.timeStep))
}
