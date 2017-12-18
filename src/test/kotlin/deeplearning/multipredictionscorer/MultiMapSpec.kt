/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.utils.MultiMap
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/**
 *
 */
class MultiMapSpec : Spek({

  describe("a MultiMap with 2 first level keys") {

    val group1 = mapOf(
      Pair(0, 10),
      Pair(1, 11),
      Pair(2, 12)
    )

    val group2 = mapOf(
      Pair(3, 13),
      Pair(4, 14)
    )

    val multimap: MultiMap<Int> = MultiMap(data = mapOf(
      Pair(5, group1),
      Pair(6, group2)
    ))

    on("get()") {

      it("should return null if getting a not existing first level key") {
        assertNull(multimap[2])
      }

      it("should return the expected first group") {
        assertEquals(multimap[5], group1)
      }

      it("should return the expected second group") {
        assertEquals(multimap[6], group2)
      }

      it("should return null if getting a not existing second level key") {
        assertNull(multimap[5, 8])
      }

      it("should return null if getting a not existing combination of keys") {
        assertNull(multimap[8, 9])
      }

      it("should return the expected existing element") {
        assertEquals(12, multimap[5, 2])
      }
    }

    on("getValue()") {

      it("should throw an exception if getting a not existing second level key") {
        assertFailsWith<KotlinNullPointerException> { multimap.getValue(5, 8) }
      }

      it("should throw an exception if getting a not existing combination of keys") {
        assertFailsWith<KotlinNullPointerException> { multimap.getValue(8, 9) }
      }

      it("should return the expected existing element") {
        assertEquals(12, multimap.getValue(5, 2))
      }
    }

    on("forEach()") {

      it("should loop over all the elements") {

        val elements = setOf(
          Pair(Pair(5, 0), 10),
          Pair(Pair(5, 1), 11),
          Pair(Pair(5, 2), 12),
          Pair(Pair(6, 3), 13),
          Pair(Pair(6, 4), 14)
        )
        val iterElements = mutableSetOf<Pair<Pair<Int, Int>, Int>>()

        multimap.forEach { i, j, elm -> iterElements.add(Pair(Pair(i, j), elm)) }

        assertEquals(elements, iterElements)
      }
    }

    on("map()") {

      val returnedMultimap = multimap.map { _, _, elm -> elm + 10}

      it("should return a MultiMap with the same first level keys") {
        assertEquals(setOf(5, 6), returnedMultimap.keys)
      }

      it("should return a MultiMap with the same second level keys of the first group") {
        assertEquals(setOf(0, 1, 2), returnedMultimap[5]!!.keys)
      }

      it("should return a MultiMap with the same second level keys of the second group") {
        assertEquals(setOf(3, 4), returnedMultimap[6]!!.keys)
      }

      it("should return a MultiMap with the expected element") {
        assertEquals(23, returnedMultimap[6, 3])
      }
    }

    on("toString()") {

      it("should return the expected string representation") {
        assertEquals("""
          {
            5: {
              0: 10,
              1: 11,
              2: 12
            },
            6: {
              3: 13,
              4: 14
            }
          }""".trimIndent(),
          multimap.toString()
        )
      }
    }
  }
})
