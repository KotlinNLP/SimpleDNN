/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import java.io.Serializable

/**
 * The [UpdatableArray] is a wrapper of an [NDArray] extending it with an [updaterSupportStructure]
 *
 */
open class UpdatableArray(open val values: NDArray<*>) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from [Serializable])
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The updater support structure to be used in combination with [com.kotlinnlp.simplednn.core.functionalities.updatemethods]
   */
  var updaterSupportStructure: UpdaterSupportStructure? = null
}
