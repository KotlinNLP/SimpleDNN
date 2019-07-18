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

  private fun findMax(paramsErrors: ParamsErrorsList): Double {

    var max = Double.MIN_VALUE
    paramsErrors.map {
      for (i in 0 until it.values.rows)
        for (j in 0 until it.values.columns)
          if (it.values[i, j].toDouble() > max)
            max = it.values[i, j].toDouble()

    }
    return max
  }

  /**
   * Clip [paramsErrors] norm, multiplying each parameter by a coefficient.
   * Coefficient is maxNorm divided by n-norm of parameters
   */
  fun clipByNorm (paramsErrors: ParamsErrorsList, maxNorm: Double, normType: String = "2") {

    var totalNorm = 0.0
    if (normType == "inf")
      totalNorm = findMax(paramsErrors)
    else {
      paramsErrors.map {
        for (i in 0 until it.values.rows)
          for (j in 0 until it.values.columns)

            totalNorm += it.values[i, j].toDouble().pow(normType.toDouble())

      }
      totalNorm = totalNorm.pow(1.0 / normType.toDouble())
    }
    val clipCoefficient = maxNorm / (totalNorm + 0.0000001)
    if (clipCoefficient < 1.0)
      paramsErrors.map {
        for (i in 0 until it.values.rows)
          for (j in 0 until it.values.columns)

            it.values[i, j] = it.values[i, j].toDouble() * clipCoefficient

    }
  }

  /**
   * Clip [paramsErrors] in-place at specified [clipValue]
   *
   * @param clipValue The gradients will be clipped at this value
   */
  fun clipByValue (paramsErrors: ParamsErrorsList, clipValue: Double) {

    paramsErrors.map {
      for (i in 0 until it.values.rows)
        for (j in 0 until it.values.columns)
          if (it.values[i, j].toDouble() < -clipValue)
            it.values[i, j] = -clipValue
          else if (it.values[i, j].toDouble() > clipValue)
            it.values[i, j] = clipValue
    }
  }
}