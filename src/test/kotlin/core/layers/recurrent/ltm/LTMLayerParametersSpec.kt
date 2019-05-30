/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class LTMLayerParametersSpec : Spek({

  describe("a LTMLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
          2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
          3.3, 3.4, 3.5, 3.6)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = LTMLayerParameters(
          inputSize = 3,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val w1 = params.inputGate1.weights.values
        val w2 = params.inputGate2.weights.values
        val w3 = params.inputGate3.weights.values
        val wCell = params.cell.weights.values

        val b1 = params.inputGate1.biases.values
        val b2 = params.inputGate2.biases.values
        val b3 = params.inputGate3.biases.values
        val bCell = params.cell.biases.values

        it("should contain the expected initialized weights of the input gate L1") {
          (0 until w1.length).forEach { i -> assertEquals(initValues[i], w1[i]) }
        }

        it("should contain the expected initialized weights of the input gate L2") {
          (0 until w2.length).forEach { i -> assertEquals(initValues[i + 9], w2[i]) }
        }

        it("should contain the expected initialized weights of the input gate L3") {
          (0 until w3.length).forEach { i -> assertEquals(initValues[i + 18], w3[i]) }
        }

        it("should contain the expected initialized weights of the cell") {
          (0 until wCell.length).forEach { i -> assertEquals(initValues[i + 27], wCell[i]) }
        }

        it("should contain the expected initialized biases of the input gate L1") {
          (0 until b1.length).forEach { i -> assertEquals(0.9, b1[i]) }
        }

        it("should contain the expected initialized biases of the input gate L3") {
          (0 until b3.length).forEach { i -> assertEquals(0.9, b3[i]) }
        }

        it("should contain the expected initialized biases of the input gate L2") {
          (0 until b2.length).forEach { i -> assertEquals(0.9, b2[i]) }
        }

        it("should contain the expected initialized biases of the cell") {
          (0 until bCell.length).forEach { i -> assertEquals(0.9, bCell[i]) }
        }
      }
    }

    context("iteration") {

      val params = LTMLayerParameters(inputSize = 3)

      val iterator = params.iterator()

      val w1 = params.inputGate1.weights
      val w2 = params.inputGate2.weights
      val w3 = params.inputGate3.weights
      val wCell = params.cell.weights

      val b1 = params.inputGate1.biases
      val b2 = params.inputGate2.biases
      val b3 = params.inputGate3.biases
      val bCell = params.cell.biases

      context("iteration 1") {
        it("should return the weights of the input gate L1") {
          assertEquals(w1, iterator.next())
        }
      }

      context("iteration 2") {
        it("should return the weights of the input gate L2") {
          assertEquals(w2, iterator.next())
        }
      }

      context("iteration 3") {
        it("should return the weights of the input gate L3") {
          assertEquals(w3, iterator.next())
        }
      }

      context("iteration 4") {
        it("should return the weights of the cell") {
          assertEquals(wCell, iterator.next())
        }
      }

      context("iteration 5") {
        it("should return the biases of the input gate L1") {
          assertEquals(b1, iterator.next())
        }
      }

      context("iteration 6") {
        it("should return the biases of the input gate L2") {
          assertEquals(b2, iterator.next())
        }
      }

      context("iteration 7") {
        it("should return the biases of the input gate L3") {
          assertEquals(b3, iterator.next())
        }
      }

      context("iteration 8") {
        it("should return the biases of the cell") {
          assertEquals(bCell, iterator.next())
        }
      }

      context("iteration 9") {
        it("should return true when calling hasNext()") {
          assertFalse(iterator.hasNext())
        }
      }
    }

    context("copy") {

      val params = LTMLayerParameters(inputSize = 3)
      val clonedParams = params.copy()

      it("should return a new element") {
        assertNotEquals(params, clonedParams)
      }

      it("should return params with same values") {
        assertTrue {
          params.zip(clonedParams).all { it.first.values.equals(it.second.values, tolerance = 1.0e-06) }
        }
      }
    }
  }
})
