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
 * @property values the values of this updatable array
 */
open class UpdatableArray<NDArrayType: NDArray<NDArrayType>>(open val values: NDArrayType) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from [Serializable])
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Build an [UpdatableArray] with values initialized to zeros.
     *
     * @param dim1 the first dimension of the array
     * @param dim2 the second dimension of the array (default = 1)
     * @param sparse whether the array is sparse or dense
     *
     * @return a new updatable array with values initialized to zeros
     */
    operator fun invoke(dim1: Int, dim2: Int = 1, sparse: Boolean = false): UpdatableArray<*> =
      if (sparse)
        UpdatableSparseArray(dim1, dim2)
      else
        UpdatableDenseArray(dim1, dim2)
  }

  /**
   * The updater support structure used by [com.kotlinnlp.simplednn.core.functionalities.updatemethods].
   */
  var updaterSupportStructure: UpdaterSupportStructure? = null

  /**
   * Return the [updaterSupportStructure].
   *
   * If the [updaterSupportStructure] is null, set it with a new [StructureType].
   * If the [updaterSupportStructure] has already been initialized, it must be compatible with the required
   * [StructureType].
   *
   * @return the [updaterSupportStructure]
   */
  inline fun <reified StructureType: UpdaterSupportStructure>getOrSetSupportStructure(): StructureType {

    if (this.updaterSupportStructure == null) {
      this.updaterSupportStructure = StructureType::class.constructors.first().call(this.values.shape)
    }

    require(this.updaterSupportStructure is StructureType) { "Incompatible support structure" }

    @Suppress("UNCHECKED_CAST")
    return this.updaterSupportStructure as StructureType
  }
}
