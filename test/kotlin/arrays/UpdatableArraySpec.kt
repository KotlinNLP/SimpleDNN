/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package arrays

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 *
 */
class UpdatableArraySpec : Spek({

  describe("an UpdatableArray") {

    context("initialization") {

      on("with the length") {

        val updatableArray = UpdatableArray(length = 5)

        it("should contain values with the expected number of rows") {
          assertEquals(5, updatableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(1, updatableArray.values.columns)
        }

        it("should contain a null support structure") {
          assertNull(updatableArray.updaterSupportStructure)
        }
      }

      on("with the shape") {

        val updatableArray = UpdatableArray(shape = Shape(3, 7))

        it("should contain values with the expected number of rows") {
          assertEquals(3, updatableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(7, updatableArray.values.columns)
        }

        it("should contain a null support structure") {
          assertNull(updatableArray.updaterSupportStructure)
        }
      }
    }
  }
})
