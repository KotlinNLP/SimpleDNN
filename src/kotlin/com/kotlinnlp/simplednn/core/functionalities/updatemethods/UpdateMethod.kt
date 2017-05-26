/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * UpdateMethod implements different gradient-based optimization algorithm (e.g. LearningRate, Adagrad, ADAM).
 *
 * @property regularization
 */
abstract class UpdateMethod(val regularization: WeightsRegularization?) {

  /**
   *
   * @param array parameter
   * @return the update neuralnetwork for the parameter
   */
  fun getSupportStructure(array: UpdatableDenseArray): UpdaterSupportStructure {

    if (array.updaterSupportStructure == null) {
      array.updaterSupportStructure = this.supportStructureFactory(array.values.shape)
    }

    require(isSupportStructureCompatible(array.updaterSupportStructure!!)){
      "Incompatible updaterSupportStructure"
    }

    return array.updaterSupportStructure!!
  }

  /**
   *
   * @param array the inputArray to update
   * @param errors errors to subtract to the inputArray, after being optimized
   */
  fun <NDArrayType: NDArray<NDArrayType>> update(array: UpdatableDenseArray, errors: NDArrayType) {

    val optimizedErrors: NDArrayType = this.optimizeErrors(errors, array)

    this.regularization?.apply(array)

    array.values.assignSub(optimizedErrors)
  }

  /**
   *
   * @param errors errors
   * @param array parameter
   * @return
   */
  abstract protected fun <NDArrayType: NDArray<NDArrayType>> optimizeErrors(
    errors: NDArrayType,
    array: UpdatableDenseArray
  ): NDArrayType

  /**
   *
   * @param shape shape
   * @return a new support structure to update an array
   */
  abstract protected fun supportStructureFactory(shape: Shape): UpdaterSupportStructure

  /**
   *
   * @param supportStructure supportStructure
   * @return Boolean
   */
  abstract protected fun isSupportStructureCompatible(supportStructure: UpdaterSupportStructure): Boolean
}
