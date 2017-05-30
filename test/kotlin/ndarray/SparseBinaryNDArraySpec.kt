/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 *
 */
class SparseBinaryNDArraySpec : Spek({

  describe("a SparseBinaryNDArray") {

    context("iteration") {

      on("row vector") {

        val array = SparseBinaryNDArrayFactory.arrayOf(activeIndices = intArrayOf(1, 3, 19), shape = Shape(1, 20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(0, 1), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(0, 3), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(0, 19), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }

      on("column vector") {

        val array = SparseBinaryNDArrayFactory.arrayOf(activeIndices = intArrayOf(1, 3, 19), shape = Shape(20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(1, 0), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(3, 0), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(19, 0), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }

      on("a 2-dim array") {

        val array = SparseBinaryNDArrayFactory.arrayOf(
          activeIndicesPairs = arrayOf(Pair(1, 0), Pair(5, 19), Pair(12, 6)),
          shape = Shape(15, 20))
        val iterator = array.iterator()

        it("should return the expected first indices Pair") {
          assertEquals(Pair(1, 0), iterator.next())
        }

        it("should return the expected second indices Pair") {
          assertEquals(Pair(5, 19), iterator.next())
        }

        it("should return the expected third indices Pair") {
          assertEquals(Pair(12, 6), iterator.next())
        }

        it("should return false calling hasNext() at the fourth iteration") {
          assertFalse { iterator.hasNext() }
        }
      }
    }
  }
})
