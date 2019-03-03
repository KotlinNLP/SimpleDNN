/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork

import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
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
class NetworkParametersSpec: Spek({

  describe("a StackedLayersParameters") {

    context("iteration over SimpleRecurrent + Feedforward parameters") {

      val params = StackedLayersParameters(listOf(
        LayerInterface(size = 3),
        LayerInterface(size = 4, connectionType = LayerType.Connection.SimpleRecurrent),
        LayerInterface(size = 2, connectionType = LayerType.Connection.Feedforward)
      ))

      on("iteration 1") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        it("should return the params of the first iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      on("iteration 2") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        iterator.next()
        firstLayerIterator.next()

        it("should return the params of the second iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      on("iteration 3") {

        val iterator = params.iterator()
        val firstLayerIterator = params.paramsPerLayer[0].iterator()

        (0 until 2).forEach { iterator.next() }
        (0 until 2).forEach { firstLayerIterator.next() }

        it("should return the params of the third iteration of the first layer") {
          assertEquals(iterator.next(), firstLayerIterator.next())
        }
      }

      on("iteration 4") {

        val iterator = params.iterator()
        val secondLayerIterator = params.paramsPerLayer[1].iterator()

        (0 until 3).forEach { iterator.next() }

        it("should return the params of the first iteration of the second layer") {
          assertEquals(iterator.next(), secondLayerIterator.next())
        }
      }

      on("iteration 5") {

        val iterator = params.iterator()
        val secondLayerIterator = params.paramsPerLayer[1].iterator()

        (0 until 4).forEach { iterator.next() }
        secondLayerIterator.next()

        it("should return the params of the second iteration of the second layer") {
          assertEquals(iterator.next(), secondLayerIterator.next())
        }
      }

      on("iteration 6") {

        val iterator = params.iterator()

        (0 until 5).forEach { iterator.next() }

        it("should return false when calling hasNext()") {
          assertFalse { iterator.hasNext() }
        }
      }
    }
  }
})
