/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The RMSProp method is a variant of the AdaGradMethod where the squared sum of previous gradients is replaced with a
 * moving average.
 *
 * @property learningRate the initial learning rate
 * @property epsilon a ias parameter
 * @property decay the rate decay parameter
 *
 * References
 * [rmsprop: Divide the gradient by a running average of its recent magnitude]
 * (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 *
 */
class RMSPropMethod(
  val learningRate: Double = 0.001,
  val epsilon: Double = 1e-08,
  val decay: Double = 0.95,
  regularization: ParamsRegularization? = null
) : UpdateMethod<RMSPropStructure>(regularization) {

  /**
   * Build a [RMSPropMethod] with a given configuration object
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.RMSPropConfig): this(
    learningRate = config.learningRate,
    epsilon = config.epsilon,
    decay = config.decay,
    regularization = config.regularization
  )

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): RMSPropStructure = array.getOrSetSupportStructure()

  /**
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized sparse errors
   */
  override fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: RMSPropStructure): SparseNDArray {

    val m = supportStructure.secondOrderMoments
    val mask: NDArrayMask = errors.mask

    m.assignProd(this.decay).assignSum(errors.prod(errors).assignProd(1.0 - this.decay))

    return errors.div(m.sqrt(mask = mask).assignSum(this.epsilon)).assignProd(this.learningRate)
  }

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  override fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: RMSPropStructure): DenseNDArray {

    val m = supportStructure.secondOrderMoments

    m.assignProd(this.decay).assignSum(errors.prod(errors).assignProd(1.0 - this.decay))

    return errors.div(m.sqrt().assignSum(this.epsilon)).assignProd(this.learningRate)
  }
}
