/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import kotlin.math.pow

/**
 * The class that performs the calculation of the gradients clipping.
 */
sealed class GradientClipping {

  /**
   * Clip the gradients in place.
   *
   * @param paramsErrors the gradients
   */
  abstract fun clip(paramsErrors: ParamsErrorsList)

  /**
   * Clip the gradients, multiplying each parameter by a coefficient.
   * Coefficient is maxNorm divided by n-norm of parameters
   *
   * @param maxNorm max norm of the gradients
   * @param normType type of the used p-norm. Can be ``Double.POSITIVE_INFINITY`` for infinity norm (default 2)
   */
  class byNorm(private val maxNorm: Double, private val normType: Double = 2.0) : GradientClipping() {

    /**
     * Clip the gradients, multiplying each parameter by a coefficient.
     * Coefficient is maxNorm divided by n-norm of parameters
     *
     * @param maxNorm max norm of the gradients
     * @param normType type of the used p-norm (default 2)
     */
    constructor(maxNorm: Double, normType: Int = 2) : this(maxNorm, normType.toDouble())

    /**
     * Clip the gradients in place.
     *
     * @param paramsErrors the gradients
     */
    override fun clip(paramsErrors: ParamsErrorsList) {

      val totalNorm: Double = if (this.normType == Double.POSITIVE_INFINITY)
        paramsErrors.map { it.values.abs().max() }.max()!!
      else {
        paramsErrors.map { it.values.abs().pow(this.normType).sum() }.sum().pow(1.0 / this.normType)
      }

      val clipCoefficient = this.maxNorm / (totalNorm + 0.0000001)

      if (clipCoefficient < 1.0) {
        paramsErrors.forEach { it.values.assignProd(clipCoefficient) }
      }
    }

  }

  /**
   * Clip gradients at specified [clipValue]
   *
   * @param clipValue the gradients will be clipped at this value
   */
  class byValue(private val clipValue: Double) : GradientClipping() {

    /**
     * Clip the gradients in place.
     *
     * @param paramsErrors the gradients
     */
    override fun clip(paramsErrors: ParamsErrorsList) {

      paramsErrors.map {
        for (i in 0 until it.values.rows) {
          for (j in 0 until it.values.columns) {
            val value = it.values[i, j].toDouble()
            when {
              value < -clipValue -> it.values[i, j] = -this.clipValue
              value > clipValue -> it.values[i, j] = this.clipValue
            }
          }
        }
      }
    }
  }
}