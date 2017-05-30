/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package ndarray

import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
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
class SparseNDArraySpec : Spek({

  describe("a SparseNDArray") {

    context("iteration") {

      on("2-dim array") {

        val array = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.1),
            Pair(Pair(1, 0), 0.5),
            Pair(Pair(1, 2), 0.1),
            Pair(Pair(2, 2), 0.2),
            Pair(Pair(3, 1), 0.3)
          ),
          shape = Shape(4, 3))
        val iterator = array.iterator()

        it("should return the expected entry on the iteration 1") {
          assertEquals(Pair(Pair(1, 0), 0.5), iterator.next())
        }

        it("should return the expected entry on the iteration 2") {
          assertEquals(Pair(Pair(0, 1), 0.1), iterator.next())
        }

        it("should return the expected entry on the iteration 3") {
          assertEquals(Pair(Pair(3, 1), 0.3), iterator.next())
        }

        it("should return the expected entry on the iteration 4") {
          assertEquals(Pair(Pair(1, 2), 0.1), iterator.next())
        }

        it("should return the expected entry on the iteration 5") {
          assertEquals(Pair(Pair(2, 2), 0.2), iterator.next())
        }

        it("should return false calling hasNext() on the last iteration") {
          assertFalse { iterator.hasNext() }
        }
      }
    }
  }
})
