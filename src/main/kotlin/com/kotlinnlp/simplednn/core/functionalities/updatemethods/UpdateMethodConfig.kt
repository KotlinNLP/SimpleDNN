/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization

/**
 * Update methods configuration.
 *
 * @property regularization a parameters regularization method
 */
sealed class UpdateMethodConfig(val regularization: ParamsRegularization? = null) {

  /**
   * AdaGrad configuration.
   *
   * @property learningRate the initial learning rate
   * @property epsilon bias parameter
   * @property regularization a parameters regularization method
   */
  class AdaGradConfig(
    val learningRate: Double = 0.01,
    val epsilon: Double = 1.0E-8,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * ADAM configuration.
   *
   * @property stepSize the initial step size
   * @property beta1 the beta1 hyper-parameter
   * @property beta2 the beta2 hyper-parameter
   * @property epsilon the epsilon hyper-parameter
   * @property regularization a parameters regularization method
   */
  class ADAMConfig(
    val stepSize: Double = 0.001,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1.0E-8,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * LearningRate configuration.
   *
   * @property learningRate the initial learning rate
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class LearningRateConfig(
    val learningRate: Double,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * Momentum configuration.
   *
   * @property learningRate the initial learning rate
   * @property momentum  the momentum
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class MomentumConfig(
    val learningRate: Double = 0.01,
    val momentum: Double = 0.9,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * NesterovMomentumConfig configuration.

   * @property learningRate the initial learning rate
   * @property momentum  the momentum
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class NesterovMomentumConfig(
    val learningRate: Double = 0.01,
    val momentum: Double = 0.9,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * RMSProp configuration.
   *
   * @property learningRate the initial learning rate
   * @property epsilon a ias parameter
   * @property decay the rate decay parameter
   * @property regularization a parameters regularization method
   */
  class RMSPropConfig(
    val learningRate: Double = 0.001,
    val epsilon: Double = 1e-08,
    val decay: Double = 0.95,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)
}
