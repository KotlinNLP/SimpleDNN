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
class GradientClipping {

  /**
   * Clip [paramsErrors] norm, multiplying each parameter by a coefficient.
   * Coefficient is maxNorm divided by n-norm of parameters
   */
  fun clipByNorm (paramsErrors: ParamsErrorsList, maxNorm: Double, normType: Int = 2) =
    this.clipByNorm(paramsErrors, maxNorm, normType.toDouble())

  /**
   * Clip [paramsErrors] norm, multiplying each parameter by a coefficient.
   * Coefficient is maxNorm divided by n-norm of parameters
   */
  fun clipByNorm (paramsErrors: ParamsErrorsList, maxNorm: Double, normType: Double = 2.0) {

    val totalNorm: Double = if (normType == Double.POSITIVE_INFINITY)
      paramsErrors.map { it.values.abs().max() }.max()!!
    else {
      paramsErrors.map { it.values.abs().pow(normType).sum() }.sum().pow(1.0 / normType)
    }

    val clipCoefficient = maxNorm / (totalNorm + 0.0000001)

    if (clipCoefficient < 1.0) {
      paramsErrors.forEach { it.values.assignProd(clipCoefficient) }
    }
  }

  /**
   * Clip [paramsErrors] in-place at specified [clipValue]
   *
   * @param clipValue The gradients will be clipped at this value
   */
  fun clipByValue (paramsErrors: ParamsErrorsList, clipValue: Double) {

    paramsErrors.map {
      for (i in 0 until it.values.rows) {
        for (j in 0 until it.values.columns) {
          val value = it.values[i, j].toDouble()
          when {
            value < -clipValue -> it.values[i, j] = -clipValue
            value > clipValue -> it.values[i, j] = clipValue
          }
        }
      }
    }
  }
}