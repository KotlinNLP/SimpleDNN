/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package arrays

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class AugmentedArraySpec : Spek({

  describe("an AugmentedArray") {

    val initArray = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val errors = NDArray.arrayOf(doubleArrayOf(0.4, 0.8, 0.1, 0.8, 0.4, 0.1, 0.2, 0.9, 0.2))

    context("initialization with an NDArray") {

      val augmentedArray = AugmentedArray(initArray)

      it("should contain values with the expected number of rows") {
        assertEquals(9, augmentedArray.values.rows)
      }

      it("should contain values with the expected number of columns") {
        assertEquals(1, augmentedArray.values.columns)
      }
    }

    context("errors") {

      on("before assignment") {

        val augmentedArray = AugmentedArray(initArray)
        val zeros = NDArray.zeros(shape = Shape(9))

        it("should have zeros errors") {
          assertEquals(true, augmentedArray.errors.equals(zeros, tolerance = 1.0e-08))
        }
      }

      on("after assignment") {

        val augmentedArray = AugmentedArray(initArray)
        augmentedArray.assignErrors(errors)

        it("should have the expected assigned errors") {
          assertEquals(true, augmentedArray.errors.equals(errors, tolerance = 1.0e-08))
        }
      }
    }

    on("cloning") {

      val augmentedArray = AugmentedArray(initArray)
      augmentedArray.setActivation(ELU())
      augmentedArray.assignErrors(errors)

      val cloneArray = augmentedArray.clone()

      it("should have the same activation function values") {
        assertEquals(augmentedArray.activationFunction, cloneArray.activationFunction)
      }

      it("should have the expected not activated values") {
        assertEquals(true, augmentedArray.valuesNotActivated.equals(cloneArray.valuesNotActivated))
      }

      it("should have the expected activated values") {
        assertEquals(true, augmentedArray.values.equals(cloneArray.values))
      }

      it("should have the expected errors") {
        assertEquals(true, augmentedArray.errors.equals(cloneArray.errors, tolerance = 1.0e-08))
      }
    }
  }
})
