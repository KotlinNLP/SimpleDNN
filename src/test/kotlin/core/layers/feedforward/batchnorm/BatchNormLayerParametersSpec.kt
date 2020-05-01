/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm.BatchNormLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals

/**
 *
 */
class BatchNormLayerParametersSpec: Spek({

  describe("BatchNormLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = BatchNormLayerParameters(
          inputSize = 10,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        it("should contain the expected initialized weights g") {
          (0 until params.g.values.length).forEach { i ->
            assertEquals(initValues[i], params.g.values[i])
          }
        }

        it("should contain the expected initialized biases b") {
          (0 until params.b.values.length).forEach { i ->
            assertEquals(0.9, params.b.values[i])
          }
        }
      }
    }
  }
})
