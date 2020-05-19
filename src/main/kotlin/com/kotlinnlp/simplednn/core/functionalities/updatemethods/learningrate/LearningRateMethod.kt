/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling

/**
 * The LearningRate method.
 *
 * @property learningRate the initial learning rate
 * @property decayMethod the rate decay method
 * @property regularization a parameters regularization method
 */
class LearningRateMethod(
  val learningRate: Double,
  val decayMethod: DecayMethod? = null,
  regularization: ParamsRegularization? = null
) : EpochScheduling, UpdateMethod<LearningRateStructure>(regularization) {

  /**
   * Build a [LearningRateMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.LearningRateConfig): this(
    learningRate = config.learningRate,
    decayMethod = config.decayMethod,
    regularization = config.regularization
  )

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): LearningRateStructure = array.getOrSetSupportStructure()

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
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized sparse errors
   */
  override fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: LearningRateStructure): SparseNDArray =
    errors.prod(this.alpha)

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  override fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: LearningRateStructure): DenseNDArray =
    supportStructure.denseErrors.assignProd(errors, this.alpha)
}
