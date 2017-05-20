/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.regularization.WeightsRegularization
import com.kotlinnlp.simplednn.simplemath.NDArray
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
  fun getSupportStructure(array: UpdatableArray): UpdaterSupportStructure {

    if (array.updaterSupportStructure == null) {
      array.updaterSupportStructure = this.supportStructureFactory(array.shape)
    }

    require(isSupportStructureCompatible(array.updaterSupportStructure as UpdaterSupportStructure)){
      "Incompatible updaterSupportStructure"
    }

    return array.updaterSupportStructure as UpdaterSupportStructure
  }

  /**
   *
   * @param array the inputArray to update
   * @param errors errors to subtract to the inputArray, after being optimized
   */
  fun update(array: UpdatableArray, errors: NDArray) {

    val optimizedErrors = this.optimizeErrors(errors, array)

    this.regularization?.apply(array)

    array.values.assignSub(optimizedErrors)
  }

  /**
   *
   * @param errors errors
   * @param array parameter
   * @return
   */
  abstract protected fun optimizeErrors(errors: NDArray, array: UpdatableArray): NDArray

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
