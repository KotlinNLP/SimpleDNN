/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import java.io.Serializable

/**
 * The ActivableArray is a superstructure of an NDArray
 *
 */
open class UpdatableArray(val shape: Shape) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Secondary constructor to create a vector of the type UpdatableArray
   */
  constructor(length: Int): this(Shape(length))

  /**
   * An NDArray containing the values of this UpdatableArray
   */
  val values: NDArray = NDArray.zeros(shape)

  /**
   * The updater support structure to be used in combination with [com.kotlinnlp.simplednn.core.functionalities.updatemethods]
   */
  var updaterSupportStructure: UpdaterSupportStructure? = null
}
