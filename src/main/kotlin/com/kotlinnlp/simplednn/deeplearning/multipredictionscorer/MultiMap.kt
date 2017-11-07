/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multipredictionscorer

/**
 * A MultiMap of [Int] indices to [Map]s of [Int] indices to generic elements.
 *
 * An example of content:
 * {
 *   1: {
 *     11: elm1
 *     12: elm2,
 *     15: elm3
 *   },
 *   5: {
 *     45: elm4,
 *     67: elm5
 *   }
 * }
 */
class MultiMap<out T>(private val data: Map<Int, Map<Int, T>>) {

  /**
   * The set of first level keys.
   */
  val keys: Set<Int> = this.data.keys

  /**
   * @param i a first level index
   *
   * @return the second level [Map] at the given index or null if not present
   */
  operator fun get(i: Int): Map<Int, T>? = this.data[i]

  /**
   * @param i a first level index
   * @param j a second level index
   *
   * @return the element at the given indices or null if not present
   */
  operator fun get(i: Int, j: Int): T? = if (this.data[i] != null) this.data[i]!![j] else null

  /**
   * @param i a first level index
   * @param j a second level index
   *
   * @return the element at the given indices, forced to be present
   */
  fun getValue(i: Int, j: Int): T = this[i, j]!!

  /**
   * Call the given callback for each element of this [MultiMap], passing the first and second level indices and the
   * related element.
   *
   * @param callback the callback function
   */
  fun forEach(callback: (i: Int, j: Int, element: T) -> Unit) {
    this.data.forEach { i, elementsMap -> elementsMap.forEach { j, element -> callback(i, j, element) } }
  }

  /**
   * Map each element of this multimap to a parallel multimap (with the same indexing), containing the results of the
   * given [transform] function, called passing the first and second level indices and the related element.
   *
   * @param transform the transform function to apply to each element
   *
   * @return a multimap of objects returned by the [transform] function
   */
  fun <OutT> map(transform: (i: Int, j: Int, element: T) -> OutT): MultiMap<OutT> {

    val keysI = this.data.keys.toList()

    return MultiMap(
      data = mapOf(*Array(
        size = keysI.size,
        init = { i ->
          val iKey: Int = keysI[i]
          val iMap: Map<Int, T> = this.data[iKey]!!
          val keysJ: List<Int> = iMap.keys.toList()

          Pair(
            iKey,
            mapOf(*Array(
              size = keysJ.size,
              init = { j ->
                val jKey: Int = keysJ[j]

                Pair(jKey, transform(iKey, jKey, iMap[jKey]!!))
              }
            ))
          )
        }
      ))
    )
  }

  /**
   * Get the string representation of this multimap with a JSON-like style.
   *
   * Example:
   * {
   *   1: {
   *     11: elm1
   *     12: elm2,
   *     15: elm3
   *   },
   *   5: {
   *     45: elm4,
   *     67: elm5
   *   }
   * }
   *
   * @return the [String] representation of this multimap
   */
  override fun toString(): String {

    val keysI: Set<Int> = this.data.keys
    val buffer = StringBuffer()

    buffer.append("{")

    keysI.forEachIndexed { iIndex, i ->

      val keysJ: Set<Int> = this.data[i]!!.keys

      buffer.append("\n  $i: {")

      keysJ.forEachIndexed { jIndex, j ->

        buffer.append("\n    $j: ${this[i, j]}")

        if (jIndex < (keysJ.size - 1)) {
          buffer.append(",")
        }
      }

      buffer.append("\n  }")

      if (iIndex < (keysI.size - 1)) {
        buffer.append(",")
      }
    }

    buffer.append("\n}")

    return buffer.toString()
  }
}
