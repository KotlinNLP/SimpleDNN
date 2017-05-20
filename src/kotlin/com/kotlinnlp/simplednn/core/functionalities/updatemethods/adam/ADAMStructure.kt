/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 * @param shape shape
 */
class ADAMStructure(shape: Shape) : UpdaterSupportStructure(shape) {

  /**
   *
   */
  val firstOrderMoments: NDArray = NDArray.zeros(shape)

  /**
   *
   */
  val secondOrderMoments: NDArray = NDArray.zeros(shape)
}
