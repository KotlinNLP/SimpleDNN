/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.concat

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class ConcatLayerStructureSpec : Spek({

  describe("a ConcatLayer") {

    on("forward") {

      val layer = ConcatLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6, 0.1, 0.0, 0.5, -0.7, -0.7, 0.8)),
            tolerance = 1.0e-05)
        }
      }
    }

    on("backward") {

      val layer = ConcatLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(ConcatLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray at index 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, -0.2)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.7)),
            tolerance = 1.0e-05)
        }
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.1, -0.7)),
            tolerance = 1.0e-05)
        }
      }
    }
  }
})
