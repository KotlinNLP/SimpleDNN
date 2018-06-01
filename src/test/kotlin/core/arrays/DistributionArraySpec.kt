/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.arrays

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.*

/**
 *
 */
class DistributionArraySpec : Spek({

  describe("an DistributionArray") {

    context("factory") {

      on("uniform distribution") {

        val array = DistributionArray.uniform(length = 10)

        it("should create an array with the expected length") {
          assertEquals(10, array.length)
        }

        it("should create an array with values uniformly distributed") {
          assertTrue { (0 until 10).all { i -> array.values[i] == 0.1 } }
        }
      }

      on("one hot distribution") {

        val array = DistributionArray.oneHot(length = 10, oneAt = 5)

        it("should create an array with the expected length") {
          assertEquals(10, array.length)
        }

        it("should contain the expected number of zeros") {
          val zerosCount = (0 until 10).count{ i -> array.values[i] == 0.0 }
          assertEquals(9, zerosCount)
        }

        it("should contain the expected 1.0 value") {
          assertEquals(1.0, array.values[5])
        }

        it("should raise an Exception if the oneAt index exceeds the length") {
          assertFails { DistributionArray.oneHot(length = 3, oneAt = 5) }
        }
      }
    }

    context("initialization") {

      val initArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, 0.2, 0.1, 0.2))
      val wrongInitArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.6, 0.2, 0.1, 0.2))
      val wrongInitArrayNeg = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, 0.2, -0.1, 0.2))

      it("should have the expected length") {
        val array = DistributionArray(values = initArray)
        assertEquals(5, array.length)
      }

      it("should raise an Exception if values are not a probability distribution") {
        assertFails { DistributionArray(values = wrongInitArray) }
      }

      it("should raise an Exception if negative values are present") {
        assertFails { DistributionArray(values = wrongInitArrayNeg) }
      }
    }

    context("cloning") {

      val array = DistributionArray.uniform(length = 10)
      val cloneArray = array.clone()

      it("should return a new array") {
        assertFalse { array === cloneArray }
      }

      it("should return an array with a new 'values' property") {
        assertFalse { array.values === cloneArray.values }
      }

      it("should return an array with equal values") {
        assertTrue { array.values.equals(cloneArray.values, tolerance = 1.0e-08) }
      }
    }
  }
})
