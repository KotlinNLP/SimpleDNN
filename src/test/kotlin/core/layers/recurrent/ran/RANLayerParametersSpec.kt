/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class RANLayerParametersSpec : Spek({

  describe("a RANLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
          1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
          1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
          2.5, 2.6)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = RANLayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val wIn = params.inputGate.weights.values
        val wFor = params.forgetGate.weights.values
        val wC = params.candidate.weights.values

        val bIn = params.inputGate.biases.values
        val bFor = params.forgetGate.biases.values
        val bC = params.candidate.biases.values

        val wInr = params.inputGate.recurrentWeights.values
        val wForr = params.forgetGate.recurrentWeights.values

        it("should contain the expected initialized weights of the input gate") {
          (0 until wIn.length).forEach { i -> assertEquals(initValues[i], wIn[i]) }
        }

        it("should contain the expected initialized weights of the forget gate") {
          (0 until wFor.length).forEach { i -> assertEquals(initValues[i + 6], wFor[i]) }
        }

        it("should contain the expected initialized weights of the candidate") {
          (0 until wC.length).forEach { i -> assertEquals(initValues[i + 12], wC[i]) }
        }

        it("should contain the expected initialized biases of the input gate") {
          (0 until bIn.length).forEach { i -> assertEquals(0.9, bIn[i]) }
        }

        it("should contain the expected initialized biases of the forget gate") {
          (0 until bFor.length).forEach { i -> assertEquals(0.9, bFor[i]) }
        }

        it("should contain the expected initialized biases of the candidate") {
          (0 until bC.length).forEach { i -> assertEquals(0.9, bFor[i]) }
        }

        it("should contain the expected initialized recurrent weights of the input gate") {
          (0 until wInr.length).forEach { i -> assertEquals(initValues[i + 18], wInr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the forget gate") {
          (0 until wForr.length).forEach { i -> assertEquals(initValues[i + 22], wForr[i]) }
        }
      }
    }
  }
})
