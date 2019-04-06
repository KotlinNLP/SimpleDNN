/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.reshape

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

class ReshapeLayerStructureSpec: Spek({

  describe("a ReshapeLayerStructure"){

    context("forward") {

      on("input size 4 x 4") {

        val layer = ReshapeLayerStructureUtils.buildLayer1()
        layer.forward()

        it("should match the expected output ") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.4, 0.1, -0.9, -0.5, -0.4, 0.3, 0.7, -0.3),
                    doubleArrayOf(0.8, 0.2, 0.6, 0.7, 0.2, -0.1, 0.6, -0.2))),
                tolerance = 1.0e-06)
          }
        }
      }
    }

    context("backward") {

      on("input size 4 x 4") {

        val layer = ReshapeLayerStructureUtils.buildLayer1()

        layer.forward()

        layer.outputArray.assignErrors(DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.5, 0.5, -0.3, 0.0, 0.6, 0.4, -0.7, 0.8),
            doubleArrayOf(0.8, 0.0, 0.1, -0.7,-0.2, 0.7, -0.9, 0.4))))

        layer.backward(propagateToInput = true)

        it("should match the expected errors of the input") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.5, 0.5, -0.3, 0.0),
                    doubleArrayOf(0.6, 0.4, -0.7, 0.8),
                    doubleArrayOf(0.8, 0.0, 0.1, -0.7),
                    doubleArrayOf(-0.2, 0.7, -0.9, 0.4))),
                tolerance = 1.0e-06)
          }
        }

      }
    }
  }
})