/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.arrays

import com.kotlinnlp.simplednn.core.functionalities.activations.*
import com.kotlinnlp.simplednn.core.arrays.ActivableArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith

/**
 *
 */
class ActivableArraySpec : Spek({

  describe("an ActivableArray") {

    val initArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val initArrayWrongSize = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1))
    val activationFunction = ELU(alpha = 1.0)
    val expectedActivatedValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(
      0.0, 0.1, 0.01, -0.09516258, -0.00995017, 1.0, 10.0, -0.63212056, -0.9999546
    ))
    val activatedValuesDeriv = DenseNDArrayFactory.arrayOf(doubleArrayOf(
      1.0, 1.0, 1.0, 0.90483742, 0.99004983, 1.0, 1.0, 0.36787944, 0.0000454
    ))

    context("initialization") {

      on("with values not assigned") {

        val activableArray = ActivableArray<DenseNDArray>(size = 9)

        it("should throw an Exception when trying to get values") {
          assertFailsWith<UninitializedPropertyAccessException> {
            activableArray.values
          }
        }
      }

      on("with the size, assigning values after") {

        val activableArray = ActivableArray<DenseNDArray>(size = 9)

        it("should throw an exception when trying to assign values with wrong size") {
          assertFails { activableArray.assignValues(initArrayWrongSize) }
        }

        activableArray.assignValues(initArray)

        it("should contain values with the expected number of rows") {
          assertEquals(9, activableArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(1, activableArray.values.columns)
        }

        it("should contain the values assigned to it") {
          assertEquals(true, activableArray.values.equals(initArray))
        }
      }

      on("with an NDArray") {

        val activableArray = ActivableArray(initArray)
        activableArray.setActivation(activationFunction)

        it("should contain the values assigned to it") {
          assertEquals(true, activableArray.values.equals(initArray))
        }
      }
    }

    on("before activation") {

      val activableArray = ActivableArray(initArray)
      activableArray.setActivation(activationFunction)

      val outActivatedValues = activableArray.getActivatedValues()

      it("should have the activated values equals to the not activated ones") {
        assertEquals(true, activableArray.values.equals(activableArray.valuesNotActivated))
      }

      it("should return an new NDArray calling getActivatedValues()") {
        assertEquals(true, outActivatedValues !== activableArray.values)
      }

      it("should return the expected activated values calling getActivatedValues()") {
        assertEquals(true, outActivatedValues.equals(expectedActivatedValues, tolerance = 1.0e-08))
      }
    }

    on("after activation") {

      val activableArray = ActivableArray(initArray)
      activableArray.setActivation(activationFunction)

      activableArray.activate()

      val outActivatedValues = activableArray.getActivatedValues()

      it("should have the expected activated values") {
        assertEquals(true, expectedActivatedValues.equals(activableArray.values, tolerance = 1.0e-08))
      }

      it("should have the expected values of the derivative of its activation") {
        assertEquals(true, activatedValuesDeriv.equals(activableArray.calculateActivationDeriv(), tolerance = 1.0e-08))
      }

      it("should return an new NDArray calling getActivatedValues()") {
        assertEquals(true, outActivatedValues !== activableArray.values)
      }

      it("should return the same activated values calling getActivatedValues()") {
        assertEquals(true, outActivatedValues.equals(activableArray.values, tolerance = 1.0e-08))
      }
    }

    on("after activated twice") {

      val activableArray = ActivableArray(initArray)
      activableArray.setActivation(activationFunction)

      val expectedActivatedValues2 = DenseNDArrayFactory.arrayOf(doubleArrayOf(
        0.0, 0.1, 0.01, -0.0907749, -0.00990083, 1.0, 10.0, -0.4685364, -0.63210386
      ))

      activableArray.activate()
      activableArray.activate()

      val outActivatedValues = activableArray.getActivatedValues()

      it("should have the expected activated values") {
        assertEquals(true, expectedActivatedValues2.equals(activableArray.values, tolerance = 1.0e-08))
      }

      it("should return the same activated values calling getActivatedValues()") {
        assertEquals(true, outActivatedValues.equals(activableArray.values, tolerance = 1.0e-08))
      }

      it("should return the previous activated values as valuesNotActivated") {
        assertEquals(true, expectedActivatedValues.equals(activableArray.valuesNotActivated, tolerance = 1.0e-08))
      }
    }

    context("cloning") {

      on("values not initialized") {

        val activableArray = ActivableArray<DenseNDArray>(size = 5)

        val cloneArray = activableArray.clone()

        it("should throw an Exception when getting values") {
          assertFailsWith<UninitializedPropertyAccessException> {
            cloneArray.values
          }
        }
      }

      on("values initialized") {

        val activableArray = ActivableArray(initArray)
        activableArray.setActivation(activationFunction)

        activableArray.activate()

        val cloneArray = activableArray.clone()

        it("should have the same not activated values") {
          assertEquals(true, activableArray.valuesNotActivated.equals(cloneArray.valuesNotActivated))
        }

        it("should have the same activated values") {
          assertEquals(true, activableArray.values.equals(cloneArray.values))
        }
      }
    }
  }
})
