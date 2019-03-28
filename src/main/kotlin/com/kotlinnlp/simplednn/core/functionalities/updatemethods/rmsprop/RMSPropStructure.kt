/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * Support structure for the [RMSPropMethod].
 *
 * @param shape the shape of the related parameter
 */
class RMSPropStructure(shape: Shape) : UpdaterSupportStructure(shape) {

  /**
   * Support array for the second order moments.
   * Its values are overridden to avoid the creation of new objects.
   */
  val secondOrderMoments: DenseNDArray = DenseNDArrayFactory.zeros(shape)
}
