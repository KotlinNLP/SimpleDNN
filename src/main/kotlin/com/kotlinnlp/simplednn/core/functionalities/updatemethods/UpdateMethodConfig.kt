/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

/**
 * Update methods configuration.
 */
sealed class UpdateMethodConfig {

  /**
   * AdaGrad configuration.
   *
   * @property learningRate Initial learning rate
   * @property epsilon Bias parameter
   */
  data class AdaGradConfig(
    val learningRate:Double = 0.01,
    val epsilon: Double = 1.0E-8
  ) : UpdateMethodConfig()

  /**
   * ADAM configuration.
   *
   * @param stepSize stepSize
   * @param beta1 beta1
   * @param beta2 beta2
   * @param epsilon epsilon
   */
  data class ADAMConfig(
    val stepSize: Double = 0.001,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1.0E-8
  ) : UpdateMethodConfig()

  /**
   * LearningRate configuration.
   *
   * @param learningRate learningRate
   */
  data class LearningRateConfig(
    val learningRate: Double
  ) : UpdateMethodConfig()

  /**
   * Momentum configuration.
   *
   * @param learningRate Double >= 0. Learning rate
   * @param momentum  Double >= 0. Parameter updates momentum
   */
  data class MomentumConfig(
    val learningRate: Double = 0.01,
    val momentum: Double = 0.9
  ) : UpdateMethodConfig()

  /**
   * NesterovMomentumConfig configuration.

   * @param learningRate Double >= 0. Learning rate
   * @param momentum  Double >= 0. Parameter updates momentum
   */
  data class NesterovMomentumConfig(
    val learningRate: Double = 0.01,
    val momentum: Double = 0.9
  ) : UpdateMethodConfig()

  /**
   * RMSProp configuration.
   *
   * @property learningRate Double >= 0. Initial learning rate
   * @property epsilon Double >= 0. Bias parameter
   * @property decay Learning rate decay parameter
   */
  data class RMSPropConfig(
    val learningRate: Double = 0.001,
    val epsilon: Double = 1e-08,
    val decay: Double = 0.95
  ) : UpdateMethodConfig()
}

